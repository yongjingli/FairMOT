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
sys.path.insert(0, '/home/liyongjing/Egolee_2021/tools')
from COLORS import RGB_COLORS


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
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

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


class MotInfer:
    def __init__(self, arch='dla_34', heads=None, head_conv=None, weights=None, K=500, ltrb=True):
        print('Start MotInfer Initializing...')
        self.arch = arch
        # heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
        # heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': 17, 'hps': 34, 'hp_offset': 2}    # mot kps model
        self.heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': 1, 'hps': 1 * 2,
                      'hp_offset': 2, 'hp_wh': 4}
        if heads is not None:
            self.heads = heads

        self.head_conv = 256
        if head_conv is not None:
            self.head_conv = head_conv

        model = create_model(self.arch, self.heads, self.head_conv)

        self.weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_dla34_coco/model_30.pth'
        if weights is not None:
            self.weights = weights
        self.device = torch.device('cuda')

        model = load_model(model, self.weights)

        # model = torch.load('/home/liyongjing/Egolee_2021/programs/FairMOT-master/local_files/repvgg_deploy.pt')
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.batch = 1
        self.top_k = K
        self.ltrb = ltrb

        self.num_kps = 1
        self.down_scale = 4

        self.img_show = None
        self.hm_inds = None
        self.hm_det = None
        self.hm_score_filter = None

        self.hps = None
        self.hm_hp_inds = None
        self.hm_hp_det = None
        print('Finish MotInfer Initialize')

    def infer_img_tensor(self, img_t):
        im_blob = torch.from_numpy(img_t).cuda().unsqueeze(0)

        img_show = img_t.transpose(1, 2, 0)
        img_show = img_show[:, :, ::-1] * 255
        img_show = img_show.astype(np.uint8).copy()
        self.img_show = img_show

        with torch.no_grad():
            output = self.model(im_blob)[-1]

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            # id_feature = output['id']
            # id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg']

            hps = output['hps']
            hm_hp = output['hm_hp'].sigmoid_()
            hp_offset = output['hp_offset']
            hp_wh = output['hp_wh']

            self.parse_hm(hm, wh, reg, hps)
            self.parse_hm_hps(hm_hp, hp_offset, hp_wh)

        # visulaize
        for i, det in enumerate(self.hm_det):
            bbox = det[0:4]
            x1, y1, x2, y2 = [int(tmp * 4) for tmp in bbox]
            color = RGB_COLORS[i % len(RGB_COLORS)]
            cv2.rectangle(self.img_show, (x1, y1), (x2, y2), color, 2)

        # for i, det in enumerate(self.hm_hp_det):
        #     bbox = det[0:4]
        #     x1, y1, x2, y2 = [int(tmp * 4) for tmp in bbox]
        #     color = (0, 255, 0)
        #     cv2.rectangle(self.img_show, (x1, y1), (x2, y2), color, 2)

        # find face of the same person
        for i, hp in enumerate(self.hps):
            hp = hp.reshape(self.num_kps, -1)
            color = RGB_COLORS[i % len(RGB_COLORS)]
            for _hp in hp:
                # _hp = hp[0]  # only show nose
                _hp = _hp.cpu().numpy()
                x = int(_hp[0] * 4)
                y = int(_hp[1] * 4)
                cv2.circle(self.img_show, (x, y), 5, color, -1)

        for i, hp in enumerate(self.hps):
            hp = hp.reshape(self.num_kps, -1)
            color = RGB_COLORS[i % len(RGB_COLORS)]
            _hp = hp[0]    # face center point
            _hp = _hp.cpu().numpy()
            x = int(_hp[0] * 4)
            y = int(_hp[1] * 4)

            face_c = np.array([x, y])
            face_boxes = np.array(self.hm_hp_det[:, 0:4] * 4)

            distance = (face_boxes[:, 0] - face_c[0]) ** 2 + (face_boxes[:, 1] - face_c[1]) ** 2
            if len(distance) > 0:
                index = np.argmin(distance)
                face_box = face_boxes[index]
                print(face_box)
                print(face_c)
                if face_box[0] < face_c[0] < face_box[2] and face_box[1] < face_c[1] < face_box[3]:
                    x1, y1, x2, y2 = [int(tmp) for tmp in face_box]
                    cv2.rectangle(self.img_show, (x1, y1), (x2, y2), color, 2)

        return self.img_show

    def parse_hm(self, hm, wh, reg, hps=None):
        hm = _nms(hm)
        batch, cat, height, width = hm.size()
        scores, hm_inds, clses, ys, xs = _topk(hm, K=self.top_k)
        if reg is not None:
            reg = _tranpose_and_gather_feat(reg, hm_inds)
            reg = reg.view(batch, self.top_k, 2)
            xs = xs.view(batch, self.top_k, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.top_k, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.top_k, 1) + 0.5
            ys = ys.view(batch, self.top_k, 1) + 0.5
        wh = _tranpose_and_gather_feat(wh, hm_inds)
        if self.ltrb:
            wh = wh.view(batch, self.top_k, 4)
        else:
            wh = wh.view(batch, self.top_k, 2)

        clses = clses.view(batch, self.top_k, 1).float()
        scores = scores.view(batch, self.top_k, 1)

        if self.ltrb:
            bboxes = torch.cat([xs - wh[..., 0:1],
                                ys - wh[..., 1:2],
                                xs + wh[..., 2:3],
                                ys + wh[..., 3:4]], dim=2)
        else:
            bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                                ys - wh[..., 1:2] / 2,
                                xs + wh[..., 0:1] / 2,
                                ys + wh[..., 1:2] / 2], dim=2)

        dets = torch.cat([bboxes, scores, clses], dim=2)

        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])

        dets = dets[0]
        score_filter = dets[:, 4] > 0.3
        dets = dets[score_filter]
        self.hm_inds = hm_inds
        self.hm_score_filter = score_filter
        self.hm_det = dets
        self.batch = batch

        if hps is not None:
            hps = _transpose_and_gather_feat(hps, self.hm_inds)
            hps = hps.view(self.batch, self.top_k, self.num_kps * 2)
            hps[..., ::2] += xs.view(self.batch, self.top_k, 1).expand(self.batch, self.top_k, self.num_kps)
            hps[..., 1::2] += ys.view(self.batch, self.top_k, 1).expand(self.batch, self.top_k, self.num_kps)

            hps = hps[0]
            hps = hps[self.hm_score_filter]
            self.hps = hps

    def parse_hm_hps(self, hm_hp, hp_offset, hp_wh):
        hm_hp = _nms(hm_hp)
        batch, cat, height, width = hm_hp.size()
        scores, inds, clses, ys, xs = _topk(hm_hp, K=self.top_k)
        if hp_offset is not None:
            hp_offset = _tranpose_and_gather_feat(hp_offset, inds)
            hp_offset = hp_offset.view(batch, self.top_k, 2)
            xs = xs.view(batch, self.top_k, 1) + hp_offset[:, :, 0:1]
            ys = ys.view(batch, self.top_k, 1) + hp_offset[:, :, 1:2]

        hp_wh = _tranpose_and_gather_feat(hp_wh, inds)
        if self.ltrb:
            hp_wh = hp_wh.view(batch, self.top_k, 4)
        else:
            hp_wh = hp_wh.view(batch, self.top_k, 2)

        clses = clses.view(batch, self.top_k, 1).float()
        scores = scores.view(batch, self.top_k, 1)

        if self.ltrb:
            bboxes = torch.cat([xs - hp_wh[..., 0:1],
                                ys - hp_wh[..., 1:2],
                                xs + hp_wh[..., 2:3],
                                ys + hp_wh[..., 3:4]], dim=2)
        else:
            bboxes = torch.cat([xs - hp_wh[..., 0:1] / 2,
                                ys - hp_wh[..., 1:2] / 2,
                                xs + hp_wh[..., 0:1] / 2,
                                ys + hp_wh[..., 1:2] / 2], dim=2)

        dets = torch.cat([bboxes, scores, clses], dim=2)

        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = dets[0]
        score_filter = dets[:, 4] > 0.3
        dets = dets[score_filter]
        self.hm_hp_inds = inds
        self.hm_hp_det = dets



def test_model_infer_torch():
    num_keys = 1
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': num_keys, 'hps': num_keys * 2, 'hp_offset': 2, 'hp_wh': 4}
    head_conv = 256
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_repvgg_b0_pose_track/model_30.pth'
    # mot_infer = MotInfer(arch='RepVGG_B0', heads=heads, head_conv=head_conv, weights=weights)

    weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_dlaconv34_pose_track/model_30.pth'
    mot_infer = MotInfer(arch='dlaconv_34', heads=heads, head_conv=head_conv, weights=weights)

    img_dir = '/home/liyongjing/Egolee_2021/data/src_person_car/2021-01-22/Person'
    # img_size = (1088, 608)
    img_size = (864, 480)
    # img_size = (576, 320)
    dataloader = datasets.LoadImages(img_dir, img_size)

    for i, (img_path, img, img0) in enumerate(dataloader):
        img_result = mot_infer.infer_img_tensor(img)

        cv2.namedWindow('img_result', 0)
        cv2.imshow('img_result', img_result)

        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            break


if __name__ == "__main__":
    # show_mot_kp_model_result()
    test_model_infer_torch() 
