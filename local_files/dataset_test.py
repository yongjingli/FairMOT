import os
import os.path as osp
import json
import numpy as np
import cv2

import sys
sys.path.insert(0, '/home/liyongjing/Egolee_2021/programs/FairMOT-master/src/lib')

sys.path.insert(0, '/home/liyongjing/Egolee_2021/programs/FairMOT-master/src/lib')
from opts import opts
from datasets.dataset_factory import get_dataset

from torchvision.transforms import transforms as T
import torch


import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../src/lib/')
from models.model import create_model, load_model
import cv2
import os
import logging
import numpy as np

import torch
import datasets.dataset.jde as datasets

import torch.nn as nn
import torch.nn.functional as F
# from .utils import _gather_feat, _tranpose_and_gather_feat
from models.utils import _gather_feat, _tranpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat



def test_dataset_transformer():
    opt = opts().parse()
    opt.data_dir = '/home/liyongjing/Egolee_2021/data/TrainData/mot_kpwh'
    # opt.data_cfg = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/src/data/coco_mot_kp.train'
    #
    Dataset = get_dataset(opt.dataset, opt.task)
    # f = open(opt.data_cfg)
    # data_config = json.load(f)
    trainset_paths = {'pose_track_mot_kpwh': '/home/liyongjing/Egolee_2021/programs/FairMOT-master/src/data/pose_track_mot_kpwh.train'}
    # dataset_root = data_config['root']
    dataset_root = opt.data_dir

    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    for iter_id, batch in enumerate(train_loader):
        imgs, labels, kps, img_path, input_size = batch
        # print('imgs shape:', imgs.shape)
        # print('labels shape:', labels.shape)
        # print('get labels:', labels)
        print('kps shape:', kps.shape)
        # print('img_path:', img_path)

        img = imgs[0].numpy()
        label = labels[0].numpy()
        kp = kps[0].numpy()

        img = img.transpose(1, 2, 0)
        img = img[:, :, ::-1] * 255
        img = img.astype(np.uint8).copy()
        h, w, _ = img.shape
        print(h, w, _)

        nL = len(label)
        if nL > 0:
            for i in range(nL):
                bbox = label[i, 2:]
                bbox[0] = bbox[0] * w
                bbox[1] = bbox[1] * h
                bbox[2] = bbox[2] * w
                bbox[3] = bbox[3] * h

                x = int(bbox[0] - bbox[2] * 0.5)
                y = int(bbox[1] - bbox[3] * 0.5)
                x2 = int(bbox[0]) + int(bbox[2] * 0.5)
                y2 = int(bbox[1]) + int(bbox[3] * 0.5)

                # bbox = [int(tmp) for tmp in bbox]
                cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 255), 2)

        nK = len(kp)
        if nK > 0:
            for i in range(nK):
                points = kp[i]
                for j in range(1):
                    point = points[j]
                    if point[2] > 0:
                        x = int(point[0] * w)
                        y = int(point[1] * h)
                        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

                        k_w = int(point[2] * w)
                        k_h = int(point[3] * h)
                        cv2.rectangle(img, (int(x - 0.5 * k_w), int(y - 0.5 * k_h)), (int(x + 0.5 * k_w), int(y + 0.5 * k_h)), (255, 0, 0))
                        # cv2.circle(img, (pts[j, 0], pts[j, 1]), 3, (255, 0, 255), -1)

        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)
        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            exit(1)


def test_dataset_load():
    opt = opts().parse()
    opt.data_dir = '/home/liyongjing/Egolee_2021/data/TrainData/mot_kpwh'
    # opt.data_cfg = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/src/data/coco_mot_kp.train'
    #
    Dataset = get_dataset(opt.dataset, opt.task)
    # f = open(opt.data_cfg)
    # data_config = json.load(f)
    trainset_paths = {'pose_track_mot_kpwh': '/home/liyongjing/Egolee_2021/programs/FairMOT-master/src/data/pose_track_mot_kpwh.train'}
    # dataset_root = data_config['root']
    dataset_root = opt.data_dir
    input_size = (1088, 608)

    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )


    K = 500
    ltrb = True

    for iter_id, batch in enumerate(train_loader):
        # print(batch.keys())
        # (['input', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'ids', 'bbox', 'hps', 'hps_mask', 'hp_offset', 'hp_ind',
        #   'hp_mask', 'hm_hp'])

        input = batch['input']
        input = input[0]
        # img = torch.from_numpy(input).cuda().unsqueeze(0)
        img_np = input.cpu().numpy()
        img_np = img_np.transpose(1, 2, 0)
        img_np = img_np[:, :, ::-1] * 255
        img_np = img_np.astype(np.uint8).copy()

        bbox = batch['bbox'][0]
        reg_mask = batch['reg_mask'][0]

        score_filter = reg_mask[:] > 0.3
        bbox = bbox[score_filter]

        num_joints = 1
        hps = batch['hps'][0]
        hps_mask = batch['hps_mask'][0]

        # show hps
        for det, hp, hp_mask in zip(bbox, hps, hps_mask):
            # show bbox
            bbox = det[0:4]
            x1, y1, x2, y2 = [int(tmp * 4) for tmp in bbox]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # show hps
            p_x = int(hp[0] * 4 + (x1 + x2) * 0.5)
            p_y = int(hp[1] * 4 + (y1 + y2) * 0.5)

            p_x_mask = hp_mask[0]
            p_y_mask = hp_mask[1]

            if p_x_mask > 0 and p_y_mask > 0:
                # print((p_x, p_y))
                cv2.circle(img_np, (p_x, p_y), 5, (0, 255, 0), -1)

        # show hm
        hm = batch['hm'].sigmoid_()
        hm = _nms(hm)
        scores, inds, clses, ys, xs = _topk(hm, K=K)
        for y, x in zip(ys[0], xs[0]):
            y = int(y * 4)
            x = int(x * 4)
            cv2.circle(img_np, (x, y), 5, (0, 0, 255), -1)


        # show hp_wh
        hp_wh = batch['hp_wh'][0]
        hp_mask = batch['hp_mask'][0]
        hp_ind = batch['hp_ind'][0]
        #
        # print('hp_wh:', hp_wh.shape)
        # print('hp_mask:', hp_mask.shape)
        # print('hp_ind:', hp_ind.shape)

        _, out_h, out_w = batch['hm'][0].shape
        # print(out_h, out_w, _)
        # exit(1)

        for _hp_wh, _mask, _hp_id in zip(hp_wh, hp_mask, hp_ind):
            if _mask:
                print(_hp_wh)
                l, t, r, b = _hp_wh * 4

                x = _hp_id.numpy() % out_w * 4
                y = _hp_id.numpy() // out_w * 4
                # print(x, y)
                x1 = int(x - l)
                y1 = int(y - t)
                x2 = int(x + r)
                y2 = int(y + b)
                # print((x1, y1), (x2, y2))
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            #
            # for i in range(len(_mask)):
            #     if _mask[i]:
            #         l, t, r, b = _hp_wh[i]
            #         x = _hp_id[i] % out_w
            #         y = _hp_id[i] // out_w
            #         x1 = x + l
            #         y1 = y + t
            #         x2 = x + r
            #         y2 = y + b
            #         cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # hm_hp hm
        # print(batch['hm_hp'].shape)
        # hm_hp = batch['hm_hp'].sigmoid_()
        # hm_hp = _nms(hm_hp)
        # scores, inds, clses, ys, xs = _topk(hm_hp, K=K)
        # for cls, y, x in zip(clses[0], ys[0], xs[0]):
        #     y = int(y * 4)
        #     x = int(x * 4)
        #     if cls == 1 or cls == 2:
        #         cv2.circle(img_np, (x, y), 5, (0, 255, 0), -1)

        cv2.namedWindow('img_np', 0)
        cv2.imshow('img_np', img_np)

        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            break


if __name__ == "__main__":
    # test_dataset_transformer()
    test_dataset_load()
