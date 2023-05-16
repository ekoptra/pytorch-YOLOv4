# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import logging
import os, sys
from tqdm import tqdm
from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
import argparse
from easydict import EasyDict as edict
from torch.nn import functional as F
from tool.utils import do_detect, load_class_names
from tool.metrics import process_batch, ap_per_class, ConfusionMatrix
import cv2
from pathlib import Path

import numpy as np
from PIL import Image

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def bboxes_giou(bboxes_a, bboxes_b, xyxy=True):
    pass


def bboxes_diou(bboxes_a, bboxes_b, xyxy=True):
    pass


def bboxes_ciou(bboxes_a, bboxes_b, xyxy=True):
    pass


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id])
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, size_average=False)
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes




def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5, save_dir=None):
    train_dataset = Yolo_dataset(config.train_label, config)
    val_dataset = Yolo_dataset(config.val_label, config)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=0,
                            pin_memory=True, drop_last=True)


    writer = SummaryWriter(log_dir=save_dir,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:                {epochs}
        Batch size:            {config.batch}
        Subdivisions:          {config.subdivisions}
        Learning rate:         {config.learning_rate}
        Training size:         {n_train}
        Checkpoints:           {save_cp}
        Device:                {device.type}
        Images size:           {config.width}
        Optimizer:             {config.TRAIN_OPTIMIZER}
        Dataset classes:       {config.classes}
        Train label path:      {config.train_label}
        Train Class Name Path: {config.class_names}
        Pretrained:
    ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate / config.batch, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions,n_classes=config.classes)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    model.train()

    for epoch in range(epochs):
        #model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step  % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    '''
                    pbar.set_postfix({'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_lr()[0] * config.batch
                                        })
                    '''
                    logging.info('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])
            
            # evaluate on last epochs or base on eval interval
            if (epoch == epochs - 1) or (((epoch + 1) % config.eval_interval) == 0):
                evaluate(model, config, writer, epoch, save_dir)

                if save_cp:
                    try:
                        checkpoint_dir = os.path.join(save_dir, config.checkpoints)
                        os.mkdir(checkpoint_dir)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'Yolov4_epoch{epoch + 1}.pth'))
                    logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def evaluate(model, config, writer, epoch, save_dir):
    eval_label = config.eval_label
    base_path = '/'.join(eval_label.split('/')[:-1])

    with open(eval_label, 'r') as f:
        val_dataset = [line.rstrip() for line in f]

    stats, ap = [], []
    seen = 0
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    class_names = load_class_names(config.class_names)

    confusion_matrix = ConfusionMatrix(nc=len(class_names))

    logging.info("\nStart to evaluate on eval dataset ...")

    width = 608
    height = 608
    for i, img_label in enumerate(val_dataset):
        parsed = img_label.split(' ')

        logging.info(f"Predict image {i+1}/{len(val_dataset)}: {parsed[0]}")
        seen += 1

        
        img = Image.open(os.path.join(base_path, parsed[0])).convert('RGB')
        img = img.resize((width, height))

        bboxes = []
        for p in parsed[1:]:
            p = p.split(',')
            bboxes.append([int(p[0]),int(p[1]), int(p[2]), int(p[3]), int(p[4])])
        

        target_bboxes = []
        for bbox in bboxes:
            target_bboxes.append([bbox[4], bbox[0], bbox[1], bbox[2], bbox[3]])


        boxes = do_detect(model, img, 0.5, cfg.classes, 0.4, 1)
        predict_bboxes = []
        for bbox in boxes:
            x1 = int((bbox[0] - bbox[2] / 2.0) * width)
            y1 = int((bbox[1] - bbox[3] / 2.0) * height)
            x2 = int((bbox[0] + bbox[2] / 2.0) * width)
            y2 = int((bbox[1] + bbox[3] / 2.0) * height)
            conf = bbox[5]
            c = bbox[6]

            predict_bboxes.append([x1, y1, x2, y2, conf, c])

        
        target_bboxes = torch.from_numpy(np.array(target_bboxes, dtype=np.float64))
        if len(predict_bboxes) == 0:
          
          stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_bboxes[:, 0].cpu()))
          continue

        predict_bboxes = torch.from_numpy(np.array(predict_bboxes, dtype=np.float64))
        
        correct = process_batch(predict_bboxes.cpu(), target_bboxes.cpu(), iouv)    
        confusion_matrix.process_batch(predict_bboxes, target_bboxes)

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), predict_bboxes[:, 4].cpu(), predict_bboxes[:, 5].cpu(), target_bboxes[:, 0].cpu()))
    
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    list_plot = []

    p, r, ap, f1, ap_class, list_plot = ap_per_class(*stats, plot=True, save_dir=save_dir, names=class_names)

    AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() -1
    logging.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, mf1, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), f1[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=len(class_names))  # number of targets per class

    # Print results
    s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
    logging.info(s)
    pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
    logging.info(pf % ('all', seen, nt.sum(), mp, mr, mf1, map50, map))

    pr_metric_result = (map50, map, mp, mr, mf1)

    # Print results per class
    for i, c in enumerate(ap_class):
        logging.info(pf % (class_names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                            f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

    confusion_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
    confusion_matrix.plot(save_dir=save_dir, names=class_names)

    list_plot.append(confusion_matrix_path)

    imgs = []
    imgs_name = []
    for img_path in list_plot:
            ori_img = cv2.imread(img_path)

            imgs.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))
    
            imgs_name.append(img_path.split('/')[-1].split('.')[0])

    for img, name in zip(imgs, imgs_name):
        try:
            writer.add_image(name, img, epoch + 1, dataformats='HWC')
        except:
            logging.warning(f'WARNING: failed to add {name} plot to tensorboard.\n')

    writer.add_scalar("metrics/mAP_0.5", pr_metric_result[0], epoch + 1)
    writer.add_scalar("metrics/mAP_0.50_0.95", pr_metric_result[1], epoch + 1)
    writer.add_scalar("metrics/P_.5iou", pr_metric_result[2], epoch + 1)
    writer.add_scalar("metrics/R_.5iou", pr_metric_result[3], epoch + 1)
    writer.add_scalar("metrics/F1_.5iou", pr_metric_result[4], epoch + 1)

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,help='Batch size', dest='batch')
    parser.add_argument('-s', '--subdivisions', metavar='S', type=int, nargs='?', default=1,help='subdivisions', dest='subdivisions')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained',type=str,default=None,help='pretrained yolov4.conv.137')
    parser.add_argument('-classes',type=int,default=80,help='dataset classes')
    parser.add_argument('-train_label_path',dest='train_label',type=str,default='train.txt',help="train label path")
    parser.add_argument('-epochs',dest='TRAIN_EPOCHS',type=int,default=10,help="number of training epochs")
    parser.add_argument('-c', '--class-names' , dest='class_names',type=str,default="_classes.txt", help="Path to name of the clasess")
    parser.add_argument('-e', '--eval-interval', dest='eval_interval', default=20, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-label-path', dest='eval_label', default='_annotations.txt', type=str, help='eval label path')
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging

def increment_name(path):
    '''increase save directory's id'''
    path = Path(path)
    sep = ''
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path


if __name__ == "__main__":
    save_dir = str(increment_name(os.path.join('runs/train', 'exp')))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging = init_logger(log_dir=save_dir)
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Yolov4(cfg.pretrained,n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, 
              save_dir=save_dir)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(save_dir, 'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
