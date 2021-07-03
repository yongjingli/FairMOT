from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}

        return loss, loss_stats


class MotLossKp(torch.nn.Module):
    def __init__(self, opt):
        super(MotLossKp, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        # self.nID = 1

        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        # self.emb_scale = 1
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        # key points loss
        self.s_kp = nn.Parameter(-1.85 * torch.ones(1))
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss()

        # self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else torch.nn.L1Loss(reduction='sum')

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        hp_loss,  hm_hp_loss, hp_offset_loss = 0, 0, 0  # key points loss

        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if len(id_target) > 0 and len(id_output) > 0:     # source code issue 205
                    id_loss += self.IDLoss(id_output, id_target)
                else:
                    id_loss += torch.tensor(0.0)

            # key points loss
            if opt.hp_weight > 0:
                hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                        batch['ind'], batch['hps']) / opt.num_stacks

            if opt.hm_hp_weight > 0:
                if not opt.mse_loss:
                    output['hm_hp'] = _sigmoid(output['hm_hp'])
                hm_hp_loss += self.crit_hm_hp(
                    output['hm_hp'], batch['hm_hp']) / opt.num_stacks

            if opt.hp_offset_weight > 0:
                hp_offset_loss += self.crit_reg(output['hp_offset'], batch['hp_mask'], batch['hp_ind'],
                                                batch['hp_offset']) / opt.num_stacks

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        kp_loss = opt.hp_weight * hp_loss + opt.hm_hp_weight * hm_hp_loss + opt.hp_offset_weight * hp_offset_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + \
               (self.s_det + self.s_id + self.s_kp) + torch.exp(-self.s_kp) * kp_loss
        loss *= 0.5

        # loss_stats = {'loss': loss, 'hm_loss': hm_loss,
        #               'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss,
                      'hp_loss': hp_loss, 'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss}    # key points loss

        return loss, loss_stats



class MotLossKpWh(torch.nn.Module):
    def __init__(self, opt):
        super(MotLossKpWh, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        # self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        # self.emb_scale = 1
        if self.nID > 1:
            self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        else:
            self.emb_scale = 1
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        # key points loss
        self.s_kp = nn.Parameter(-1.85 * torch.ones(1))
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss()
        # self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else torch.nn.L1Loss(reduction='sum')

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        hp_loss,  hm_hp_loss, hp_offset_loss, hp_wh_loss = 0, 0, 0, 0  # key points loss

        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                if len(id_target) > 0 and len(id_output) > 0:     # source code issue 205
                    id_loss += self.IDLoss(id_output, id_target)
                else:
                    id_loss += torch.tensor(0.0)

            # key points loss
            if opt.hp_weight > 0:
                hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                        batch['ind'], batch['hps']) / opt.num_stacks
            else:
                hp_loss += torch.tensor(0.0)

            if opt.hm_hp_weight > 0:
                if not opt.mse_loss:
                    output['hm_hp'] = _sigmoid(output['hm_hp'])
                hm_hp_loss += self.crit_hm_hp(
                    output['hm_hp'], batch['hm_hp']) / opt.num_stacks
            else:
                hm_hp_loss += torch.tensor(0.0)

            if opt.hp_offset_weight > 0:
                hp_offset_loss += self.crit_reg(output['hp_offset'], batch['hp_mask'], batch['hp_ind'],
                                                batch['hp_offset']) / opt.num_stacks
            else:
                hp_offset_loss += torch.tensor(0.0)

            if opt.hp_wh_weight > 0:
                hp_wh_loss += self.crit_reg(output['hp_wh'], batch['hp_mask'], batch['hp_ind'],
                                                batch['hp_wh']) / opt.num_stacks
            else:
                hp_wh_loss += torch.tensor(0.0)


        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        kp_loss = opt.hp_weight * hp_loss + opt.hm_hp_weight * hm_hp_loss + opt.hp_offset_weight * hp_offset_loss + opt.hp_wh_weight * hp_wh_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + \
               (self.s_det + self.s_id + self.s_kp) + torch.exp(-self.s_kp) * kp_loss
        loss *= 0.5

        # loss_stats = {'loss': loss, 'hm_loss': hm_loss,
        #               'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss,
                      'hp_loss': hp_loss, 'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss, 'hp_wh_loss': hp_wh_loss}    # key points loss

        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


class MotKpTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotKpTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'hp_loss', 'hm_hp_loss', 'hp_offset_loss']
        loss = MotLossKp(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


class MotKpWhTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotKpWhTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'hp_loss', 'hm_hp_loss', 'hp_offset_loss', 'hp_wh_loss']
        loss = MotLossKpWh(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
          
