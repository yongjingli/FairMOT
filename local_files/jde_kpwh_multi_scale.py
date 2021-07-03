import glob
import math
import os
import os.path as osp
import random
import time

from collections import OrderedDict, defaultdict

import cv2
import json
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, height, width):
        # img_path = '/home/liyongjing/Egolee_2021/data/TrainData/mot_kp/coco_mot_kp/images/000000387655.jpg'
        # label_path = '/home/liyongjing/Egolee_2021/data/TrainData/mot_kp/coco_mot_kp/labels_with_ids/000000387655.txt'

        # height = self.height
        # width = self.width

        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)
        #
        # # Load labels
        if os.path.isfile(label_path):
            # labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            only_track = False
            with open(label_path, 'r') as f:
                ss = f.readlines()
                if len(ss) > 0:
                    label_num = len(ss[0].rstrip().split(' '))
                    if label_num != (6 + self.num_joints * 5):
                        only_track = True

            if not only_track:
                assert self.num_joints == 1
                label_len = 6 + self.num_joints * 5
                labels_roi = np.loadtxt(label_path, dtype=np.float32).reshape(-1, label_len)
            else:
                label_len = 6 + 0* 5
                labels_roi = np.loadtxt(label_path, dtype=np.float32).reshape(-1, label_len)

                fill_zero = np.zeros((labels_roi.shape[0], 5), dtype=np.float32)
                labels_roi = np.concatenate((labels_roi, fill_zero), axis=1)


            labels0 = labels_roi[:, 0:6]
            kps0 = labels_roi[:, 6:]
            # kps0 = kps0.reshape(-1, self.num_joints, 5)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh

            #
            kps = kps0.copy()
            # kps[:, :, 0] = ratio * w * (kps[:, :, 0] - kps[:, :, 2] / 2) + padw
            # kps[:, :, 1] = ratio * h * (kps[:, :, 1] - kps[:, :, 3] / 2) + padh
            #
            # kps[:, :, 2] = ratio * w * (kps[:, :, 0] + kps[:, :, 2] / 2) + padw
            # kps[:, :, 3] = ratio * h * (kps[:, :, 1] + kps[:, :, 3] / 2) + padh

            kps[:, 0] = ratio * w * (kps0[:, 0] - kps0[:, 2] / 2) + padw
            kps[:, 1] = ratio * h * (kps0[:, 1] - kps0[:, 3] / 2) + padh
            kps[:, 2] = ratio * w * (kps0[:, 0] + kps0[:, 2] / 2) + padw
            kps[:, 3] = ratio * h * (kps0[:, 1] + kps0[:, 3] / 2) + padh

        else:
            labels = np.array([])
            kps = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M, kps = random_affine(img, labels, kps,  degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height

        nK = len(kps)
        if nK > 0:
            # kps[:, :, 0] = kps[:, :, 0] / width
            # kps[:, :, 1] = kps[:, :, 1] / height
            # convert xyxy to xywh
            kps[:, 0:4] = xyxy2xywh(kps[:, 0:4].copy())  # / height
            kps[:, 0] /= width
            kps[:, 1] /= height
            kps[:, 2] /= width
            kps[:, 3] /= height

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

                if nK > 0:
                    # kps[:, :, 0] = 1 - kps[:, :, 0]
                    kps[:, 0] = 1 - kps[:, 0]

        kps = kps.reshape(-1, self.num_joints, 5)
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, kps, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def random_affine(img, targets=None, kps=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            assert len(targets) == len(kps)

            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]


            # ========================key points========================
            n = kps.shape[0]
            points = kps[:, 0:4].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            # w = xy[:, 2] - xy[:, 0]
            # h = xy[:, 3] - xy[:, 1]
            # area = w * h
            # ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            # i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)
            kps = kps[i]
            kps[:, 0:4] = xy[i]
            # kps[:, 0:4] = xy

        return imw, targets, M, kps
    else:
        return imw


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)

# # ---------- Predefined multi-scale input image width and height list
# Input_WHs = [
#     [640, 320],   # 0
#     [672, 352],   # 1
#     [704, 384],   # 2
#     [736, 416],   # 3
#     [768, 448],   # 4
#     [800, 480],   # 5
#     [832, 512],   # 6
#     [864, 544],   # 7
#     [896, 576],   # 8
#     [928, 608],   # 9
#     [960, 640],   # 10
#     [992, 672],   # 11
#     [1064, 704],  # 12
#     [1064, 736],  # 13
#     [1064, 608],  # 14
#     [1088, 608]   # 15
# ]  # total 16 scales with floating aspect ratios

# ---------- Predefined multi-scale input image width and height list

Input_WHs = [
    [288, 160],   # 0
    [360, 200],
    [432, 240],
    [504, 280],
    [576, 320],   # 1
    [648, 360],
    [720, 400],
    [792, 440],
    [864, 480],   # 2
]

class JointDatasetKpWhMultiScale(LoadImagesAndLabels):  # for training
    print("use JointDatasetKpWhMultiScale dataset")
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None, num_joints=1):
        # print('use JointDatasetKpWhs')
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1
        self.num_joints = num_joints

        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]
        # print(self.label_files)
        # exit(1)
        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

        # for multi scale
        # define mapping from batch idx to scale idx
        self.batch_i_to_scale_i = defaultdict(int)
        self.input_multi_scales = None

        if 1:  # whether to generate multi-scales while keeping aspect ratio
            self.gen_multi_scale_input_whs()

        # rand scale the first time
        self.rand_scale()
        print('Total {:d} multi-scales:\n'.format(len(self.input_multi_scales)), self.input_multi_scales)

    def gen_multi_scale_input_whs(self, num_scales=256, min_ratio=0.5, max_ratio=1.0):
        """
        generate input multi scale image sizes(w, h), keep default aspect ratio
        :param num_scales:
        :return:
        """
        gs = 32  # grid size

        self.input_multi_scales = [x for x in Input_WHs if not (x[0] % gs or x[1] % gs)]
        self.input_multi_scales.append([self.width, self.height])

        # ----- min scale and max scale
        # keep default aspect ratio
        self.default_aspect_ratio = self.height / self.width

        # min scale
        min_width = math.ceil(self.width * min_ratio / gs) * gs
        min_height = math.ceil(self.height * min_ratio / gs) * gs
        self.input_multi_scales.append([min_width, min_height])

        # max scale
        max_width = math.ceil(self.width * max_ratio / gs) * gs
        max_height = math.ceil(self.height * max_ratio / gs) * gs
        self.input_multi_scales.append([max_width, max_height])

        # other scales
        # widths = list(range(min_width, max_width + 1, int((max_width - min_width) / num_scales)))
        # heights = list(range(min_height, max_height + 1, int((max_height - min_height) / num_scales)))
        widths = list(range(min_width, max_width + 1, 1))
        heights = list(range(min_height, max_height + 1, 1))
        widths = [width for width in widths if not (width % gs)]
        heights = [height for height in heights if not (height % gs)]
        if len(widths) < len(heights):
            for width in widths:
                height = math.ceil(width * self.default_aspect_ratio / gs) * gs
                if [width, height] in self.input_multi_scales:
                    continue
                self.input_multi_scales.append([width, height])
        elif len(widths) > len(heights):
            for height in heights:
                width = math.ceil(height / self.default_aspect_ratio / gs) * gs
                if [width, height] in self.input_multi_scales:
                    continue
                self.input_multi_scales.append([width, height])
        else:
            for width, height in zip(widths, heights):
                if [width, height] in self.input_multi_scales:
                    continue
                height = math.ceil(width * self.default_aspect_ratio / gs) * gs
                self.input_multi_scales.append([width, height])

        if len(self.input_multi_scales) < 2:
            self.input_multi_scales = None
            print('[warning]: generate multi-scales failed(keeping aspect ratio)')
        else:
            self.input_multi_scales.sort(key=lambda x: x[0])

    def rand_scale(self):
        # randomly generate mapping from batch idx to scale idx
        if self.input_multi_scales is None:
            self.num_batches = self.nF // self.opt.batch_size + 1
            for batch_i in range(self.num_batches):
                rand_batch_idx = np.random.randint(0, self.num_batches)
                rand_scale_idx = rand_batch_idx % len(Input_WHs)
                self.batch_i_to_scale_i[batch_i] = rand_scale_idx
        else:
            self.num_batches = self.nF // self.opt.batch_size + 1
            for batch_i in range(self.num_batches):
                rand_batch_idx = np.random.randint(0, self.num_batches)
                rand_scale_idx = rand_batch_idx % len(self.input_multi_scales)
                self.batch_i_to_scale_i[batch_i] = rand_scale_idx

    def __getitem__(self, files_index):
        batch_i = files_index // int(self.opt.batch_size)
        scale_idx = self.batch_i_to_scale_i[batch_i]
        if self.input_multi_scales is None:
            width, height = Input_WHs[scale_idx]
        else:
            width, height = self.input_multi_scales[scale_idx]

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, keypoints, img_path, (input_h, input_w) = self.get_data(img_path, label_path, height, width)
        # print(keypoints)

        # return imgs, labels, keypoints, img_path, (input_h, input_w)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        num_objs_kps = keypoints.shape[0]
        assert num_objs == num_objs_kps, 'label path {}'.format(label_path) + 'num_objs:{}'.format(num_objs) + 'num_objs_kps:{}'.format(num_objs_kps)

        # Get center net labels
        # det and reid
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)

        # 'ids', reid only need 'ids'
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        # key points, beside above , need to add 'hps', 'hps_mask', 'hm_hp', 'hp_offset', 'hp_ind', 'hp_mask' # key points
        num_joints = self.num_joints
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)           # 'hps'
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)   # 'hps_mask'
        hm_hp = np.zeros((num_joints, output_h, output_w), dtype=np.float32)    # 'hm_hp'

        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)     # 'hp_offset'
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)             # 'hp_ind'
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)            # 'hp_mask'

        if self.opt.ltrb:
            hp_wh = np.zeros((self.max_objs * num_joints, 4), dtype=np.float32)
        else:
            hp_wh = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_xy

            # key point GT
            pts = keypoints[k]

            for j in range(num_joints):
                # w = pts[j, 2]
                # h = pts[j, 3]
                # area = w * h
                # ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                # filter_kp = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

                if pts[j, 4] > 0:
                # if pts[j, 4] > 0 and filter_kp:
                    pts[j, 0] = pts[j, 0] * output_w
                    pts[j, 1] = pts[j, 1] * output_h
                    pts[j, 0] = np.clip(pts[j, 0], 0, output_w - 1)
                    pts[j, 1] = np.clip(pts[j, 1], 0, output_h - 1)

                    # kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
                    # kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
                    # hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
                    #
                    # hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
                    # hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
                    # hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

                    if 0 <= pts[j, 0] < output_w and output_h > pts[j, 1] >= 0:
                        kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int   # key points offset from center
                        kps_mask[k, j * 2: j * 2 + 2] = 1   # kps mask

                        # used to get key points position, and no wh regress
                        pt_int = pts[j, :2].astype(np.int32)  # kp int center
                        # kp center offset, different between int and float
                        hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                        hp_ind[k * num_joints + j] = pt_int[1] * output_w + pt_int[0]
                        hp_mask[k * num_joints + j] = 1
                        draw_gaussian(hm_hp[j], pt_int, radius)

                        # use to get keypoints width and height
                        pts[j, 2] = pts[j, 2] * output_w
                        pts[j, 3] = pts[j, 3] * output_h
                        k_h = pts[j, 2]
                        k_w = pts[j, 3]
                        # print(pts[j, 0], pts[j, 1])
                        ct_kp = np.array([pts[j, 0], pts[j, 1]], dtype=np.float32)

                        if self.opt.ltrb:
                            kp_bbox_amodal = copy.deepcopy(pts[j, :4])
                            kp_bbox_amodal[0] = kp_bbox_amodal[0] - kp_bbox_amodal[2] / 2.
                            kp_bbox_amodal[1] = kp_bbox_amodal[1] - kp_bbox_amodal[3] / 2.
                            kp_bbox_amodal[2] = kp_bbox_amodal[0] + kp_bbox_amodal[2]
                            kp_bbox_amodal[3] = kp_bbox_amodal[1] + kp_bbox_amodal[3]
                            hp_wh[k * num_joints + j] = ct_kp[0] - kp_bbox_amodal[0], ct_kp[1] - kp_bbox_amodal[1], \
                                    kp_bbox_amodal[2] - ct_kp[0], kp_bbox_amodal[3] - ct_kp[1]
                        else:
                            hp_wh[k * num_joints + j] = 1. * k_w, 1. * k_h

        # ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys,
               'hps': kps, 'hps_mask': kps_mask, 'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask, 'hm_hp': hm_hp, 'hp_wh': hp_wh}
        return ret


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, labels0, img_path, (h, w) 
