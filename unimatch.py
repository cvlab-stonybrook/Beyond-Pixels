import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.semseg.maxpool_label import Poolinglabel
from model.semseg.maxpool_label import calculate_receptive_field
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)#local-rank(exxact),local_rank
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    poolinglabel_model = Poolinglabel(cfg['nclass'])
    poolinglabel_model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_multilabel=nn.BCEWithLogitsLoss(reduction='none').cuda(local_rank)

    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_l) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    #calcualate the size of the original image
    t_img, t_mask = trainset_l[0]##[3, 801, 801]) torch.Size([801, 801]
    org_u_left=0
    org_v_right= t_mask.shape[1]-1
    pred_u_left =0
    pred_v_right=200
    list_receptive_field=[]
    for st in range(pred_v_right+1):
        list_receptive_field.append([calculate_receptive_field(st,st)])
    receptive_pad_left = org_u_left-list_receptive_field[0][0][0]
    receptive_pad_right = list_receptive_field[-1][0][1]-org_v_right
    receptive_kernel_size = list_receptive_field[0][0][1]-list_receptive_field[0][0][0]
    receptive_stride = list_receptive_field[1][0][0]-list_receptive_field[0][0][0]
    weigth_avgpool = nn.AvgPool2d(kernel_size=receptive_kernel_size, stride=receptive_stride, padding=receptive_pad_left, count_include_pad=False)
    temp_map = torch.zeros(4,81,81)
    for _i_ in range(81):
        for _j_ in range(81):
            _u_x,_v_x = calculate_receptive_field(_i_,_i_)
            if(_u_x<0):
                _u_x=0
            if(_u_x>=321):
                _u_x=320
            if(_v_x<0):
                _v_x=0
            if(_v_x>=321):
                _v_x=320            
            temp_map[0,_i_,_j_]=_u_x
            temp_map[1,_i_,_j_]=_v_x
            _u_y,_v_y = calculate_receptive_field(_j_,_j_)
            if(_u_y<0):
                _u_y=0
            if(_u_y>=321):
                _u_y=320
            if(_v_y<0):
                _v_y=0
            if(_v_y>=321):
                _v_y=320            
            temp_map[2,_i_,_j_]=_u_y
            temp_map[3,_i_,_j_]=_v_y

    temp_map = temp_map.view(temp_map.shape[0],-1)
    temp_map = temp_map.permute(1,0)#[40401, 4]
    all_class_delta_rel = {}
    for _class_ in range(cfg['nclass']):
        all_class_delta_rel[_class_]=0.00001
    all_class_labeled_unconf={}
    for _class_ in range(cfg['nclass']):
        all_class_labeled_unconf[_class_]=AverageMeter(length=4)
    all_class_max_labeled_unconf = {}
    for _class_ in range(cfg['nclass']):
        all_class_max_labeled_unconf[_class_]=0    
    
    
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_loss_multi = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_l, trainloader_l)
        for i, ((img_x, mask_x), (_,_), (_,_)) in enumerate(loader):
            #print(mask_x.shape)#[1, 321, 321]
            with torch.no_grad():
                mask_patches = torch.stack([mask_x==i for i in range(cfg['nclass'])], dim=1)#1,19,801,801
                mask_patches = mask_patches.float().cuda()
                mask_patches = weigth_avgpool(mask_patches)#[1, 19, 201, 201]
            mask_patches = mask_patches.cuda()
            model.train()            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            num_lb, num_ulb = img_x.shape[0], img_x.shape[0]
            c1ss,_,preds, _ = model(torch.cat((img_x, img_x)), need_fp=True, classify=True, nlabel=num_lb)
            pred_x, _ = preds.split([num_lb, num_ulb])#[1, 21, 321, 321]
            c1s_x, _ = c1ss.split([num_lb, num_ulb])#[1, 6561, 21]
            c1s_x_prob = torch.sigmoid(c1s_x)
            c1s_x_prob = c1s_x_prob.detach()
            with torch.no_grad():
                all_multilabel = poolinglabel_model(mask_x)
                all_multilabel = all_multilabel.cuda().detach()
                all_multilabel = all_multilabel.view(all_multilabel.shape[0],all_multilabel.shape[1],-1)
                all_multilabel = all_multilabel.permute(0,2,1)#[1, 6561, 21]
            mask_patches = mask_patches.view(mask_patches.shape[0], mask_patches.shape[1],-1)
            mask_patches = mask_patches.permute(0,2,1)#[1, 40401, 19]
            mask_patches = F.softmax(mask_patches, dim=2)            
            loss_multi_x = criterion_multilabel(c1s_x, all_multilabel)#[1, 40401, 19]
            loss_multi_x = loss_multi_x * mask_patches#[1, 40401, 19]
            loss_multi_x = torch.sum(loss_multi_x)/torch.sum(mask_patches)
            loss_x = criterion_l(pred_x, mask_x)
            loss = loss_x+loss_multi_x
            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_multi.update(loss_multi_x.item())
            iters = epoch * len(trainloader_l) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_multi', loss_multi_x.item(), iters)
            
            if (i % (len(trainloader_l) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss multi_x: {:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_multi.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, str(epoch)+'_latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
