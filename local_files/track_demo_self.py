from multitracker_self import JDETrackerLocal
import torch

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../src/lib/')
from models.model import create_model, load_model
import cv2
import os
import logging
import numpy as np

from src.lib.tracking_utils.timer import Timer
from src.lib.tracking_utils.log import logger
logger.setLevel(logging.INFO)

from tracking_utils import visualization as vis
import datasets.dataset.jde as datasets


if __name__ == '__main__':
    print('Start...')
    # create model
    device = torch.device('cuda')

    print('Creating model...')
    # arch = 'dla_34'
    arch = 'RepVGG_B0'
    # arch = 'dlaconv_34'
    # heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
    heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2, 'hm_hp': 1, 'hps': 1 * 2, 'hp_offset': 2, 'hp_wh': 4}
    head_conv = 256
    model = create_model(arch, heads, head_conv)
    # weights = '/home/liyongjing/Egolee_2021/programs/FairMOT-master/models/fairmot_dla34.pth'
    weights = "/home/liyongjing/Egolee_2021/programs/FairMOT-master/exp/mot_kpwh/mot_kpwh_repvgg_b0_only_person_0519/model_last.pth"

    model = load_model(model, weights)
    model = model.to(device)
    model.eval()


    # self, model, track_buffer = 30, conf_thres = 0.4, k = 500, num_classes = 1, down_ratio = 4, reg_offset = True, frame_rate = 30, ltrb = True
    tracker = JDETrackerLocal(model, conf_thres=0.4)

    # start tracker
    video_path = '/home/liyongjing/Egolee_2021/data/src_track/1-20210510_15-00-16_cut.mp4'
    # video_path = "/home/liyongjing/Egolee_2021/data/src_track/1-20210510_15-00-16.mp4"
    # video_path = '/home/liyongjing/Egolee_2021/tools/video2.mp4'
    # video_path = '/home/liyongjing/Egolee_2021/tools/cut_video.mp4'
    # video_path = '/home/liyongjing/Egolee_2021/data/src_person_car/person_car.mp4'
    save_dir = '/home/liyongjing/Egolee_2021/data/src_track/track_result_imgs'
    # img_size = (1088, 608)
    img_size = (864, 480)
    # img_size = (576, 320)
    dataloader = datasets.LoadVideo(video_path, img_size)

    timer = Timer()
    results = []
    frame_id = 0

    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        # print('111')
        if i % 5 != 0:
            continue
        # logger.info('Starting tracking...')

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            # exit(1)

        # run tracking
        timer.tic()
        if True:      #use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            min_box_area = 100
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

        # print(img0.shape)
        save_path = './roi_imgs/'
        for j, online_tlwh in enumerate(online_tlwhs):
            x, y, w, h = [int(tmp) for tmp in online_tlwh]
            img_roi = img0[y:y+h, x:x+w, :]
            img_roi_name = save_path + str(i) + '_' + str(j) + '.jpg'
            if not img_roi.all():
                cv2.imwrite(img_roi_name, img_roi)
        # exit(1)

        show_image = True
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.namedWindow('online_im', 0)
            cv2.resizeWindow('online_im', 864, 480)
            cv2.imshow('online_im', online_im)
        # print(save_dir)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        frame_id += 1
        wait_key = cv2.waitKey(1)
        if wait_key == 27:
            exit(1)
