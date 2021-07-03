import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../src/lib/')
from models.model import create_model, load_model

import datasets.dataset.jde as datasets
import torch
import numpy as np
import cv2
import os
import os.path as osp

from models.networks.pose_dla_repvgg_conv import DLASeg #.DLASeg as repvgg_dla_seg
from models.networks.repvgg import whole_model_convert



def whole_model_repvgg_convert():
    num_keys = 1
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': num_keys, 'hps': num_keys * 2, 'hp_offset': 2, 'hp_wh': 4}

    head_conv = 256
    arch = 'RepVGG_B0'
    weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_pose_track_leakly/model_30.pth'
    model = create_model(arch, heads, head_conv)
    model = load_model(model, weights)
    model.eval()    # import

    # deploy model
    base_name = arch
    pretrained = False
    down_ratio = 4
    final_kernel = 1
    last_level = 5
    head_conv = head_conv

    deploy_model = DLASeg(base_name, heads, pretrained, down_ratio, final_kernel, last_level,
                          head_conv=head_conv, deploy=True)
    deploy_model.eval()

    for name, module in (model.named_children()):
        if name == 'base':
            print('base'*5)
            model_base = model.base
            # print(model_base.state_dict().keys())
            # exit(1)

            converted_weights = {}
            for name, module in model_base.named_modules():
                if hasattr(module, 'repvgg_convert'):
                    kernel, bias = module.repvgg_convert()
                    converted_weights[name + '.rbr_reparam.weight'] = kernel
                    converted_weights[name + '.rbr_reparam.bias'] = bias
                elif isinstance(module, torch.nn.Linear):
                    # converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
                    # converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
                    converted_weights[name + '.weight'] = module.weight
                    converted_weights[name + '.bias'] = module.bias

            # for name, param in deploy_model.base.named_parameters():
            #     # print('deploy param: ', name, param.size(), np.mean(converted_weights[name]))
            #     param.data = torch.from_numpy(converted_weights[name]).float()

            deploy_model.base.load_state_dict(converted_weights, strict=True)
            # deploy_model.base = deploy_model_base

            # deploy_model.base.lo
            # d
        else:
            setattr(deploy_model, name, module)

    # img_dir = '/home/liyongjing/Egolee_2021/data/src_person_car/2021-01-22/Person'
    # img_size = (1088, 608)
    img_size = (864, 480)
    # img_size = (576, 320)
    # dataloader = datasets.LoadImages(img_dir, img_size)
    #
    # for i, (img_path, img, img0) in enumerate(dataloader):
    #     im_blob = torch.from_numpy(img).unsqueeze(0)
    #
    #     img_show = img.transpose(1, 2, 0)
    #     img_show = img_show[:, :, ::-1] * 255
    #     img_show = img_show.astype(np.uint8).copy()

        # with torch.no_grad():
        #     output = model.base(im_blob)[-1]
        #     deploy_output = deploy_model.base(im_blob)[-1]
        #     print(output.shape)
        #     print(deploy_output.shape)
        #     print(output[0][0, 0:, 12:20])
        #     print(deploy_output[0][0, 0:, 12:20])
        #
        # exit(1)

        # with torch.no_grad():
        #     # output = model(im_blob)[-1]
        #     # print(output['hm'])
        #
        #     deploy_output = deploy_model(im_blob)[-1]
        #     print(deploy_output['hm'])
        #     hm = deploy_output['hm'].sigmoid_()
        #     print(hm)
        #
        #
        #
        #     wh = deploy_output['wh']
        #     # id_feature = output['id']
        #     # id_feature = F.normalize(id_feature, dim=1)
        #     reg = deploy_output['reg']
        #
        #     hps = deploy_output['hps']
        #     hm_hp = deploy_output['hm_hp'].sigmoid_()
        #     hp_offset = deploy_output['hp_offset']
        #     hp_wh = deploy_output['hp_wh']
        #
        # torch.save(deploy_model, 'repvgg_deploy.pt')

    batch_size = 1
    img = torch.randn(batch_size, 3, img_size[1], img_size[0])
    model.to(torch.device('cpu'))
    # img.to(device)
    y = model(img)

    import onnx
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    # f = 'mot_repvgg_deploy.onnx'
    f = weights.split('.')[0] + '_deploy' + '.onnx'

    # for _y in y[-1]:
    #     print(y)
    # exit(1)

    # torch.onnx.export(deploy_model, img, f, verbose=False, opset_version=9, input_names=['images'],
    #                   output_names=heads.keys())

    torch.onnx.export(deploy_model, img, f, verbose=False, opset_version=9, input_names=['images'],
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

    # exit(1)



    # cv2.namedWindow('img_show', 0)
    # cv2.imshow('img_show', img_show)
    #
    # wait_key = cv2.waitKey(0)
    # if wait_key == 27:
    #     break

# def test_repvgg_conc():
#     train_model = create_RepVGG_A0(deploy=False)
#     train_model.eval()  # Don't forget to call this before inference.
#     deploy_model = repvgg_model_convert(train_model, create_RepVGG_A0)
#     x = torch.randn(1, 3, 224, 224)
#     train_y = train_model(x)
#     deploy_y = deploy_model(x)
#     print(((train_y - deploy_y) ** 2).sum())  # Will be around 1e-10




if __name__ == '__main__':
    whole_model_repvgg_convert() 
