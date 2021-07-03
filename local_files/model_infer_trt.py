import onnxruntime as rt
import numpy as np
import cv2
import sys
import time
import os
import os.path as osp

sys.path.insert(0, '/home/liyongjing/Egolee_2021/tools')
from COLORS import RGB_COLORS

import tensorrt as trt   # tensorRT 7.0
import pycuda.driver as cuda
import pycuda.autoinit

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # set batch size


def build_engine_trt(model_file, max_batch_size=1):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
            network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        builder.max_batch_size = max_batch_size
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, name=None):
        self.host = host_mem
        self.device = device_mem
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def GiB(val):
    return val * 1 << 30


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        bindig_shape = tuple(engine.get_binding_shape(binding))
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(bindig_shape, dtype)
        # print('\tAllocate host buffer: host_mem -> {}, {}'.format(host_mem, host_mem.nbytes))  # host mem

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # print('\tAllocate device buffer: device_mem -> {}, {}'.format(device_mem, int(device_mem))) # device mem

        # print('\t# Append the device buffer to device bindings.......')
        bindings.append(int(device_mem))
        # print('\tbindings: ', bindings)

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            # print('____HostDeviceMem(host_mem, device_mem)): {}, {}'.format(HostDeviceMem(host_mem, device_mem),type(HostDeviceMem(host_mem, device_mem))))
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            # print("This is the output!")
            outputs.append(HostDeviceMem(host_mem, device_mem, binding))
        # print("----------------------end allocating one binding in the onnx model-------------------------")

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    # return [out.host for out in outputs]
    return {out.name: out.host for out in outputs}


class MotBox:
    def __init__(self, x, y, w, h, score, cls):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x1 = int(max(0, x - 0.5 * w))
        self.y1 = int(max(0, y - 0.5 * h))

        self.x2 = int(x + 0.5 * w)
        self.y2 = int(y + 0.5 * h)

        self.score = score
        self.cls = cls

        self.id_features = None

        self.has_hps = False
        self.hps_x = None
        self.hps_y = None

        self.has_box_hps = False
        self.box_hps_x = None
        self.box_hps_y = None
        self.box_hps_w = None
        self.box_hps_h = None
        self.box_hps_x1 = None
        self.box_hps_y1 = None

        self.box_hps_x2 = None
        self.box_hps_y2 = None

        self.box_hps_dist = None


class FairTrtInfer:
    def __init__(self, onnx_file, batch_size=1, min_score=0.4, net_w=None, net_h=None):
        print('FairTrt Initial Start...')
        self.onnx_file = onnx_file
        # self.sess = rt.InferenceSession(onnx_file)
        # self.input_name = self.sess.get_inputs()[0].name

        self.engine = build_engine_trt(onnx_file, batch_size)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

        self.net_w = 864
        self.net_h = 480
        if net_w is not None:
            self.net_w = net_w
        if net_h is not None:
            self.net_h = net_h

        # self.net_w = 224
        # self.net_h = 64

        self.batch_size = batch_size
        self.down_scale = 4
        self.min_score = min_score

        self.img_show = None
        self.batch_mot_boxes = None
        self.batch_hps_boxes = None
        self.show = False

        print('FairTrt Initial Done...')

    def infer_cv_img(self, cv_img):
        img_input, pad_info = self.img_pre_process(cv_img)
        mot_boxes = self.get_trt_prediction(img_input)

        if self.show:
            for i, mot_box in enumerate(mot_boxes[0]):
                color = RGB_COLORS[i % len(RGB_COLORS)]
                x = mot_box.x
                y = mot_box.y
                x1 = mot_box.x1
                y1 = mot_box.y1
                x2 = mot_box.x2
                y2 = mot_box.y2
                cv2.circle(self.img_show, (x, y), 10, color, -1)
                cv2.rectangle(self.img_show, (x1, y1), (x2, y2), color, 2)
                if mot_box.has_box_hps:
                    box_hps_x1 = mot_box.box_hps_x1
                    box_hps_y1 = mot_box.box_hps_y1

                    box_hps_x2 = mot_box.box_hps_x2
                    box_hps_y2 = mot_box.box_hps_y2
                    cv2.rectangle(self.img_show, (box_hps_x1, box_hps_y1), (box_hps_x2, box_hps_y2), color, 2)

            cv2.namedWindow('img_show', 0)
            cv2.imshow('img_show', self.img_show)

        scale_mot_boxes = self.scale_boxes(pad_info)

        return scale_mot_boxes

    def img_pre_process(self, cv_img):
        assert cv_img is not None
        h_scale = self.net_h/cv_img.shape[0]
        w_scale = self.net_w/cv_img.shape[1]

        scale = min(h_scale, w_scale)
        img_temp = cv2.resize(cv_img, (int(cv_img.shape[1] * scale), int(cv_img.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)

        # cal pad_w, and pad_h
        pad_h = (self.net_h - img_temp.shape[0]) // 2
        pad_w = (self.net_w - img_temp.shape[1]) // 2
        pad_top, pad_bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        pad_left, pad_right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        img_input = np.ones((self.net_h, self.net_w, 3), dtype=np.float32) * 127.5
        # img_input[self.pad_h:img_temp.shape[0] + self.pad_h, self.pad_w:img_temp.shape[1] + self.pad_w, :] = img_temp
        img_input[pad_top:img_temp.shape[0] + pad_bottom, pad_left:img_temp.shape[1] + pad_right, :] = img_temp

        # set img show
        self.img_show = img_input.astype(np.uint8)

        # Convert
        img_input = img_input.astype(np.float32)
        img_input = img_input[:, :, ::-1]
        img_input /= 255.0
        img_input = img_input.transpose(2, 0, 1)  # to C, H, W
        img_input = np.ascontiguousarray(img_input)
        img_input = np.expand_dims(img_input, 0)

        return img_input, [scale, pad_top, pad_bottom, pad_left, pad_right]

    def get_trt_prediction(self, input_img):
        np.copyto(self.inputs[0].host, input_img)
        feats = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                             stream=self.stream, batch_size=self.batch_size)

        # hm = feats[0]  # do sigmoid
        # wh = feats[1]
        # id = feats[2]
        # reg = feats[3]
        # hm_hp = feats[4]
        # hps = feats[5]
        # hp_offset = feats[6]
        # hp_wh = feats[7]

        hm = feats['hm']  # do sigmoid
        wh = feats['wh']
        id = feats['id']
        reg = feats['reg']
        hm_hp = feats['hm_hp']
        hps = feats['hps']
        hp_offset = feats['hp_offset']
        hp_wh = feats['hp_wh']

        self.parse_hm(hm, reg, wh, hps, id)
        self.parse_hm_hp(hm_hp, hp_offset, hp_wh)

        self.associate_hps()
        return self.batch_mot_boxes

    def parse_hm(self, hm, reg, wh, hps, id):
        hm_v = hm[:, 0:1, :, :]
        hm_maxpool = hm[:, 1:2, :, :]

        keep = (hm_maxpool == hm_v)
        keep = keep.astype(np.float)
        hm_v = hm_v * keep

        batch, *_ = hm_v.shape
        self.batch_mot_boxes = []
        for _batch in range(batch):
            _hm_v = hm_v[_batch]
            _reg = reg[_batch]
            _wh = wh[_batch]
            _hps = hps[_batch]
            _id = id[_batch]

            mot_boxes = self._parse_hm(_hm_v, _reg, _wh, _hps, _id)
            self.batch_mot_boxes.append(mot_boxes)

    def parse_hm_hp(self, hm_hp, hp_offset, hp_wh):
        hm_hp_v = hm_hp[:, 0:1, :, :]
        hm_hp_maxpool = hm_hp[:, 1:2, :, :]

        keep = (hm_hp_maxpool == hm_hp_v)
        keep = keep.astype(np.float)
        hm_hp_v = hm_hp_v * keep

        batch, *_ = hm_hp_v.shape
        self.batch_hps_boxes = []
        for _batch in range(batch):
            _hm_hp_v = hm_hp_v[_batch]
            _hp_offset = hp_offset[_batch]
            _hp_wh = hp_wh[_batch]

            hps_boxes = self._parse_hm(_hm_hp_v, _hp_offset, _hp_wh, None)
            self.batch_hps_boxes.append(hps_boxes)

    def _parse_hm(self, _hm_v, _reg, _wh, _hps=None, _id_feature=None):
        mot_boxes = []
        inds = np.where(_hm_v > self.min_score)
        for _c, _h, _w in zip(inds[0], inds[1], inds[2]):
            score = _hm_v[_c, _h, _w]
            cls = _c

            x = int(_w + _reg[0, _h, _w])
            y = int(_h + _reg[1, _h, _w])

            x1 = int(x - _wh[0, _h, _w])
            y1 = int(y - _wh[1, _h, _w])
            x2 = int(x + _wh[2, _h, _w])
            y2 = int(y + _wh[3, _h, _w])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            x = x * self.down_scale
            y = y * self.down_scale
            w = w * self.down_scale
            h = h * self.down_scale

            mot_box = MotBox(x, y, w, h, score, cls)
            # cv2.circle(self.img_show, (x, y), 10, (255, 0, 255), -1)
            # cv2.rectangle(self.img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if _id_feature is not None:
                # id_feat = _id_feature[:, _h, _w]
                id_feat = _id_feature[_h, _w, :]
                mot_box.id_feat = id_feat

            if _hps is not None:
                hps_x = int(_hps[0, _h, _w] + _w)
                hps_y = int(_hps[1, _h, _w] + _h)

                hps_x = hps_x * self.down_scale
                hps_y = hps_y * self.down_scale
                mot_box.has_hps = True
                mot_box.hps_x = hps_x
                mot_box.hps_y = hps_y
                # cv2.circle(self.img_show, (hps_x, hps_y), 10, (0, 0, 255), -1)

            mot_boxes.append(mot_box)
        return mot_boxes

    def associate_hps(self):
        batch_size = len(self.batch_mot_boxes)
        for _batch_size in range(batch_size):
            for mot_box in self.batch_mot_boxes[_batch_size]:
                for hps_box in self.batch_hps_boxes[_batch_size]:
                    if mot_box.has_hps:
                        hps_x = mot_box.hps_x
                        hps_y = mot_box.hps_y

                        hps_box_x = hps_box.x
                        hps_box_y = hps_box.y
                        hps_box_x1 = hps_box.x1
                        hps_box_y1 = hps_box.y1
                        hps_box_x2 = hps_box.x2
                        hps_box_y2 = hps_box.y2

                        hps_box_w = hps_box.w
                        hps_box_h = hps_box.h

                        if hps_box_x1 < hps_x < hps_box_x2 and hps_box_y1 < hps_y < hps_box_y2:
                            center_dist = (hps_x - hps_box_x) ** 2 + (hps_y - hps_box_y) ** 2
                            if mot_box.has_box_hps:
                                if center_dist < mot_box.box_hps_dist:
                                    mot_box.box_hps_dist = center_dist
                                    mot_box.box_hps_x = hps_box_x
                                    mot_box.box_hps_y = hps_box_y
                                    mot_box.box_hps_x1 = hps_box_x1
                                    mot_box.box_hps_y1 = hps_box_y1
                                    mot_box.box_hps_x2 = hps_box_x2
                                    mot_box.box_hps_y2 = hps_box_y2

                                    mot_box.box_hps_w = hps_box_w
                                    mot_box.box_hps_h = hps_box_h

                            else:
                                mot_box.has_box_hps = True
                                mot_box.box_hps_dist = center_dist
                                mot_box.box_hps_x = hps_box_x
                                mot_box.box_hps_y = hps_box_y
                                mot_box.box_hps_x1 = hps_box_x1
                                mot_box.box_hps_y1 = hps_box_y1
                                mot_box.box_hps_x2 = hps_box_x2
                                mot_box.box_hps_y2 = hps_box_y2
                                mot_box.box_hps_w = hps_box_w
                                mot_box.box_hps_h = hps_box_h

    def scale_boxes(self, pad_info=[1, 0, 0, 0, 0]):
        scale, pad_top, pad_bottom, pad_left, pad_right = pad_info
        batch_size = len(self.batch_mot_boxes)
        for _batch_size in range(batch_size):
            for mot_box in self.batch_mot_boxes[_batch_size]:
                mot_box.x = int((mot_box.x - pad_left) / scale)
                mot_box.y = int((mot_box.y - pad_top) / scale)

                mot_box.w = int(mot_box.w / scale)
                mot_box.h = int(mot_box.h / scale)

                mot_box.x1 = int((mot_box.x1 - pad_left) / scale)
                mot_box.y1 = int((mot_box.y1 - pad_top) / scale)

                mot_box.x2 = int((mot_box.x2 - pad_left) / scale)
                mot_box.y2 = int((mot_box.y2 - pad_top) / scale)

                if mot_box.has_hps:
                    mot_box.hps_x = int((mot_box.hps_x - pad_left) / scale)
                    mot_box.hps_y = int((mot_box.hps_y - pad_top) / scale)

                if mot_box.has_box_hps:
                    mot_box.box_hps_x = int((mot_box.box_hps_x - pad_left) / scale)
                    mot_box.box_hps_y = int((mot_box.box_hps_y - pad_top) / scale)

                    mot_box.box_hps_w = int(mot_box.box_hps_w / scale)
                    mot_box.box_hps_h = int(mot_box.box_hps_h / scale)

                    mot_box.box_hps_x1 = int((mot_box.box_hps_x1 - pad_left) / scale)
                    mot_box.box_hps_y1 = int((mot_box.box_hps_y1 - pad_top) / scale)

                    mot_box.box_hps_x2 = int((mot_box.box_hps_x2 - pad_left) / scale)
                    mot_box.box_hps_y2 = int((mot_box.box_hps_y2 - pad_top) / scale)
        return self.batch_mot_boxes

    def get_infer_feats(self, img):
        img_input, pad_info = self.img_pre_process(img)

        np.copyto(self.inputs[0].host, img_input)
        feats = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                             stream=self.stream, batch_size=self.batch_size)
        return feats


def test_fairOnnx():
    # onnx_path = '/home/liyongjing/Egolee/programs/GoroboAIReason/models/face_det_with_track/face_det_with_track.onnx'

    # onnx_path = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/local_files/mot_repvgg_deploy_sim.onnx'
    # onnx_path = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_pose_track/model_15_sim.onnx'
    # onnx_path = '/home/liyongjing/Egolee_2021/programs/RepVGG-main/trained_model/RepVggB2-crop-short-size/RepVGG-B2-deploy_sim.onnx'
    # onnx_path = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/local_files/fairmot_sim.onnx"
    onnx_path = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_pose_track_leakly/model_30_deploy_sim.onnx'
    fair_trt_infer = FairTrtInfer(onnx_path)

    # img_dir = '/home/liyongjing/Egolee_2021/data/src_person_car/2021-01-22/Person'
    img_dir = '/home/liyongjing/Egolee_2021/data/TrainData/mot_kpwh/pose_track_mot_kpwh_val/images'
    img_names = [img_name for img_name in os.listdir(img_dir) if osp.splitext(img_name)[-1] in ['.jpg']]
    for img_name in img_names:
        img_path = osp.join(img_dir, img_name)
        img_path = "/home/liyongjing/Egolee_2021/data/TrainData/mot_kpwh/pose_track_mot_kpwh_val/images/val_016423_mpii_test_000000.jpg"

        img = cv2.imread(img_path)

        s_time = time.time()
        mot_boxes = fair_trt_infer.infer_cv_img(img)
        infer_time = time.time() - s_time
        print('infer time {}s:'.format(infer_time))

        for i, mot_box in enumerate(mot_boxes[0]):
            color = RGB_COLORS[i % len(RGB_COLORS)]
            x = mot_box.x
            y = mot_box.y
            x1 = mot_box.x1
            y1 = mot_box.y1
            x2 = mot_box.x2
            y2 = mot_box.y2
            cv2.circle(img, (x, y), 10, color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            print(mot_box.score)
            if mot_box.has_box_hps:
                box_hps_x1 = mot_box.box_hps_x1
                box_hps_y1 = mot_box.box_hps_y1

                box_hps_x2 = mot_box.box_hps_x2
                box_hps_y2 = mot_box.box_hps_y2
                cv2.rectangle(img, (box_hps_x1, box_hps_y1), (box_hps_x2, box_hps_y2), color, 2)

        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)

        wait_key = cv2.waitKey(0)
        if wait_key == 27:
            exit(1)


if __name__ == '__main__':
    test_fairOnnx()
