import onnx

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../src/lib/')
from models.model import create_model, load_model
import torch
import cv2
import os
import logging


from src.lib.tracking_utils.timer import Timer
from src.lib.tracking_utils.log import logger
logger.setLevel(logging.INFO)

from tracking_utils import visualization as vis
import datasets.dataset.jde as datasets

from types import MethodType
from torch.onnx import OperatorExportTypes
import torch.nn as nn

#
# def pose_dla_forward(self, x):
#     x = self.base(x)
#     x = self.dla_up(x)
#     y = []
#     for i in range(self.last_level - self.first_level):
#         y.append(x[i].clone())
#     self.ida_up(y, 0, len(y))
#     ret = []  ## change dict to list
#     z = {}
#     for head in self.heads:
#         z[head] = self.__getattr__(head)(y[-1])
#         if head in ['hm',  'hm_hp']:
#             z[head] = z[head].sigmoid_()
#             kernel = 3
#             pad = (kernel - 1) // 2
#             hmax = nn.functional.max_pool2d(z[head], (kernel, kernel), stride=1, padding=pad)
#             z[head] = torch.cat([z[head], hmax], 1)
#         if head in 'id':
#             # x = x.div(x.norm(p=2, dim=1, keepdim=True))
#             z[head] = z[head].div(z[head].norm(p=2, dim=1, keepdim=True))
#             # z[head] = F.normalize(z[head], dim=1)
#             z[head] = z[head].permute(0, 2, 3, 1)
#
#         ret.append(z[head])
#         # ret.append(self.__getattr__(head)(y[-1]))
#     return ret
#
#
#
#
#
# ## for dla34v0
# def dlav0_forward(self, x):
#     x = self.base(x)
#     x = self.dla_up(x[self.first_level:])
#     # x = self.fc(x)
#     # y = self.softmax(self.up(x))
#     ret = []  ## change dict to list
#     z = {}
#     for head in self.heads:
#         z[head] = self.__getattr__(head)(y[-1])
#         if head in ['hm',  'hm_hp']:
#             z[head] = z[head].sigmoid_()
#             kernel = 3
#             pad = (kernel - 1) // 2
#             hmax = nn.functional.max_pool2d(z[head], (kernel, kernel), stride=1, padding=pad)
#             z[head] = torch.cat([z[head], hmax], 1)
#         if head in 'id':
#             # x = x.div(x.norm(p=2, dim=1, keepdim=True))
#             z[head] = z[head].div(z[head].norm(p=2, dim=1, keepdim=True))
#             # z[head] = F.normalize(z[head], dim=1)
#             z[head] = z[head].permute(0, 2, 3, 1)
#
#         ret.append(z[head])
#         # ret.append(self.__getattr__(head)(y[-1]))
#     return ret
#
# ## for resdcn
# def resnet_dcn_forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     x = self.deconv_layers(x)
#     z = {}
#
#     for head in self.heads:
#         z[head] = self.__getattr__(head)(y[-1])
#         if head in ['hm',  'hm_hp']:
#             z[head] = z[head].sigmoid_()
#             kernel = 3
#             pad = (kernel - 1) // 2
#             hmax = nn.functional.max_pool2d(z[head], (kernel, kernel), stride=1, padding=pad)
#             z[head] = torch.cat([z[head], hmax], 1)
#         if head in 'id':
#             # x = x.div(x.norm(p=2, dim=1, keepdim=True))
#             z[head] = z[head].div(z[head].norm(p=2, dim=1, keepdim=True))
#             # z[head] = F.normalize(z[head], dim=1)
#             z[head] = z[head].permute(0, 2, 3, 1)
#
#         ret.append(z[head])
#         # ret.append(self.__getattr__(head)(y[-1]))
#     return ret
#
#
#
# def repvgg_forward(self, x):
#     x = self.base(x)
#     x = self.dla_up(x)
#
#     y = []
#     for i in range(self.last_level - self.first_level):
#         y.append(x[i].clone())
#     self.ida_up(y, 0, len(y))
#
#     z = {}
#     ret = []
#     for head in self.heads:
#         z[head] = self.__getattr__(head)(y[-1])
#         if torch.onnx.is_in_onnx_export():
#             if head in ['hm',  'hm_hp']:
#                 z[head] = z[head].sigmoid_()
#                 kernel = 3
#                 pad = (kernel - 1) // 2
#                 hmax = nn.functional.max_pool2d(z[head], (kernel, kernel), stride=1, padding=pad)
#                 z[head] = torch.cat([z[head], hmax], 1)
#             if head in 'id':
#                 # x = x.div(x.norm(p=2, dim=1, keepdim=True))
#                 z[head] = z[head].div(z[head].norm(p=2, dim=1, keepdim=True))
#                 # z[head] = F.normalize(z[head], dim=1)
#                 z[head] = z[head].permute(0, 2, 3, 1)
#         ret.append(z[head])
#
#     return ret
#
#
# forward = {'dla':pose_dla_forward,'dlav0':dlav0_forward,'resdcn':resnet_dcn_forward, 'RepVGG':repvgg_forward}


def model_export():
    print('Creating model...')
    arch = 'dla_34'
    weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/models/fairmot_dla34.pth.old_version'


    # arch = 'hrnet_18'
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/models/hrnetv2_w18_imagenet_pretrained.pth'


    # arch = 'dlaconv_34'
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_dlaconv34_pose_track/model_30.pth'

    # # #
    # arch = 'RepVGG_B0'
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_repvgg_b0_finetune/model_last.pth.old_version'
    #

    # heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': 1, 'hps': 1 * 2, 'hp_offset': 2, 'hp_wh': 4}
    head_conv = 256

    model = create_model(arch, heads, head_conv)
    # model.forward = MethodType(forward[arch.split('_')[0]], model)

    model = load_model(model, weights)

    # device = torch.device('cuda')
    # model = model.to(device)
    model.eval()
    model.cuda()

    # export setting
    # Input
    # img_size = (1088, 608)
    # img_size = (864, 480)
    img_size = (576, 320)
    batch_size = 1
    img = torch.randn(batch_size, 3, img_size[1], img_size[0]).cuda()
    # img.to(device)
    with torch.no_grad():
        y = model(img)

    # # test op
    # base = model.base
    # test_shapes = base(img)
    # for test_shape in test_shapes:
    #     print(test_shape.shape)
    # exit(1)



    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    # f = 'fairmot.onnx'
    f = weights.split('.')[0] + '.onnx'
    # torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
    #                   output_names=y[-1].keys())

    torch.onnx.export(model, img, f, verbose=False, operator_export_type=OperatorExportTypes.ONNX, input_names=['images'],
                      output_names=heads.keys())

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('====ONNX export success, saved as %s' % f)

    # simpily onnx
    from onnxsim import simplify
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    f2 = f.replace('.onnx', '_sim.onnx')  # filename
    onnx.save(model_simp, f2)
    print('====ONNX SIM export success, saved as %s' % f2)

    # # check output different between pytorch and onnx: y, y_onnx
    # import onnxruntime as rt
    # input_all = [node.name for node in onnx_model.graph.input]
    # input_initializer = [node.name for node in onnx_model.graph.initializer]
    # net_feed_input = list(set(input_all) - set(input_initializer))
    # assert (len(net_feed_input) == 1)
    # sess = rt.InferenceSession(f2)
    # y_onnx = sess.run(None, {net_feed_input[0]: img.detach().numpy()})
    #
    # for i, (_y, _y_onnx) in enumerate(zip(y, y_onnx)):
    #     _y_numpy = _y.detach().numpy()
    #     # all_close = np.allclose(_y_numpy, _y_onnx, rtol=1e-05, atol=1e-06)
    #     diff = _y_numpy - _y_onnx
    #     print('output {}:, max diff {}'.format(i, np.max(diff)))
    #     # assert(np.max(diff) > 1e-5)
    #
    # from onnx import shape_inference
    # f3 = f2.replace('.onnx', '_shape.onnx')  # filename
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f2)), f3)
    # print('====ONNX shape inference export success, saved as %s' % f3)


if __name__ == '__main__':
    model_export() 
