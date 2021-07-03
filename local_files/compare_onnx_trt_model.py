from model_infer_onnx import FairOnnxInfer
from model_infer_trt  import FairTrtInfer
import cv2
import numpy as np


def test_compare_onnx_trt_result():
    # net_w = 768
    # net_h = 768
    # onnx_path = "/home/liyongjing/Egolee/programs/GoroboAIReason/models/car_plate_reg/car_plate_mbv3.onnx_sim_model.onnx"

    # net_w = 224
    # net_h = 64
    # onnx_path = "/home/liyongjing/Egolee/programs/GoroboAIReason/models/car_plate_reg/plate_reg.onnx_sim_model.onnx"

    net_w = 864
    net_h = 480
    onnx_path = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_pose_track_leakly/model_30_sim.onnx'

    # onnx_path = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_dlaconv34_pose_track/model_30.onnx"
    # onnx_path = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kp_repvgg_b0_finetune/model_30_sim.onnx"
    # onnx_path = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_pose_track_no_dla/model_30_sim.onnx"

    # onnx_path = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot/vggface2_hrnet18/model_last_sim.onnx'

    # net_w = 1920
    # net_h = 1056
    # onnx_path = '/home/liyongjing/Egolee_2021/programs/TensorRT-CenterNet-master/model/centerface.onnx'

    fair_onnx_infer = FairOnnxInfer(onnx_path, net_w=net_w, net_h=net_h)
    fair_trt_infer = FairTrtInfer(onnx_path, net_w=net_w, net_h=net_h)

    img_path = "/home/liyongjing/Egolee_2021/data/src_track/pose_track_labelme/train/014235_mpii_train/000037.jpg"
    img = cv2.imread(img_path)

    onnx_feats = fair_onnx_infer.get_infer_feats(img)
    trt_feats = fair_trt_infer.get_infer_feats(img)

    # onnx_feats.sort(key=lambda x: x.shape[2], reverse=True)
    # trt_feats.sort(key=lambda x: x.shape[2], reverse=True)
    print(onnx_feats.keys())
    print(trt_feats.keys())
    for output_name in onnx_feats.keys():
        onnx_feat = onnx_feats[output_name]
        trt_feat = trt_feats[output_name]
        assert onnx_feat.shape == trt_feat.shape
        diff = onnx_feat - trt_feat
        max_diff = np.max(diff)
        print('output: {}, shape{}, max diff:{}'.format(output_name, onnx_feat.shape, max_diff))
    #
    # for i, (onnx_feat, trt_feat) in enumerate(zip(onnx_feats, trt_feats)):
    #     assert onnx_feat.shape == trt_feat.shape
    #     diff = onnx_feat - trt_feat
    #     max_diff = np.max(diff)
    #     print('{} input, shape{}, max diff:{}'.format(i, onnx_feat.shape, max_diff))
    #
    # # assert onnx_feats[0].shape == trt_feats[0].shape
    # assert onnx_feats[1].shape[1] == 4
    # assert onnx_feats[7].shape[1] == 4
    # assert trt_feats[1].shape[1] == 4
    # assert trt_feats[7].shape[1] == 4
    #
    # diff_1 = onnx_feats[1] - trt_feats[1]
    # print("diff 1:", np.max(diff_1))
    #
    # # assert onnx_feats[0].shape == trt_feats[4].shape
    # diff_2 = onnx_feats[1] - trt_feats[7]
    # print("diff 2:", np.max(diff_2))


if __name__ == "__main__":
    test_compare_onnx_trt_result() 
