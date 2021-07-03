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

def save_pt_to_old_version():
    model_path = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_repvgg_b0_finetune/model_last.pth"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    new_model_path = model_path + ".old_version"
    torch.save(checkpoint, new_model_path,_use_new_zipfile_serialization=False)
    print(checkpoint.keys())



def model_export():
    print('Creating model...')

    # heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
    # head_conv = 256
    #
    arch = 'dla_34'
    weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/models/fairmot_dla34.pth'


    # arch = 'hrnet_18'
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/models/hrnetv2_w18_imagenet_pretrained.pth'


    # arch = 'dlaconv_34'
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_dlaconv_34_only_person/model_last.pth'

    # #
    # arch = 'RepVGG_B0'
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_only_person/model_last.pth'
    #

    # arch = 'hrnet_18'
    # weights = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot/vggface2_hrnet18/model_last.pth"

    # heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': 1, 'hps': 1 * 2, 'hp_offset': 2, 'hp_wh': 4}
    head_conv = 256

    model = create_model(arch, heads, head_conv)

    model = load_model(model, weights)

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    # export setting
    # Input
    # img_size = (1088, 608)
    img_size = (864, 480)
    batch_size = 1
    img = torch.randn(batch_size, 3, img_size[1], img_size[0])
    img = img.to(device)

    with torch.no_grad():
        y = model(img)

    # # test op
    # base = model.base
    # test_shapes = base(img)
    # for test_shape in test_shapes:
    #     print(test_shape.shape)
    # exit(1)


    import onnx

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    # f = 'fairmot.onnx'
    f = weights.split('.')[0] + '.onnx'
    # torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
    #                   output_names=y[-1].keys())
    # torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
    #                   output_names=heads.keys())
    torch.onnx.export(model, img, f, verbose=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX, input_names=['images'],
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
    # save_pt_to_old_version() 
