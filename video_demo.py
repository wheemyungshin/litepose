# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import json
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader, make_train_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from arch_manager import ArchManager

import mmcv
import cv2
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
                        
    #fixed config for supernet
    parser.add_argument('--superconfig',
                        default=None,
                        type=str,
                        help='fixed arch for supernet training')

    parser.add_argument('--video-path', type=str, default=None, help='Video path (video file or dir)')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--resize-w',
        type=int,
        default=0)
    parser.add_argument(
        '--resize-h',
        type=int,
        default=0)
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.2)



    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    # change the resolution according to config
    fixed_arch = None
    if args.superconfig is not None:
        with open(args.superconfig, 'r') as f:
           fixed_arch = json.load(f)
        cfg.defrost()
        reso = fixed_arch['img_size']
        cfg.DATASET.INPUT_SIZE = reso
        cfg.DATASET.OUTPUT_SIZE = [reso // 4, reso // 2]
        cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'pose_mobilenet' or cfg.MODEL.NAME == 'pose_simplenet':
        arch_manager = ArchManager(cfg)
        cfg_arch = arch_manager.fixed_sample()
        if fixed_arch is not None:
            cfg_arch = fixed_arch
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True, cfg_arch = cfg_arch
        )
    else:
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )

    #set super config
    if cfg.MODEL.NAME == 'pose_supermobilenet':
        model.arch_manager.is_search = True
        if args.superconfig is not None:
            with open(args.superconfig, 'r') as f:
                model.arch_manager.search_arch = json.load(f)
        else:
            model.arch_manager.search_arch = model.arch_manager.fixed_sample()

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(cfg.DATASET.INPUT_SIZE, model, dump_input))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # load video
    
    video_path_list = []
    if os.path.isdir(args.video_path):
        print("Video Path is Dir.")
        for video_file in os.listdir(args.video_path):
            video_path = os.path.join(args.video_path, video_file)
            video_path_list.append(video_path)
    else:
        video_path_list = [args.video_path]
   
    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    #eval mode
    model.eval()

    # read video
    for video_path in video_path_list:
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Faild to load video file {video_path}'

        if args.out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_video_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = video.fps
            if args.resize_h == 0 or args.resize_w == 0:
                size = (video.width, video.height)
            else:
                size = (args.resize_w, args.resize_h)
            
            print("SIZE: ", size)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root,
                            f'vis_{os.path.basename(video_path)}'), fourcc,
                fps, size)
        print("Loading...:", video_path)

        for i, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            if not (args.resize_h == 0 or args.resize_w == 0):
                cur_frame = cv2.resize(cur_frame, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)

            #image = images.cpu().numpy()
            # size at scale 1.0
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
            )

            with torch.no_grad():
                infer_begin = time.time()
                final_heatmaps = None
                tags_list = []
                for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                    input_size = cfg.DATASET.INPUT_SIZE
                    image_resized, center, scale = resize_align_multi_scale(
                        image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                    )
                    image_resized = transforms(image_resized)
                    image_resized = image_resized.unsqueeze(0).cuda()

                    outputs, heatmaps, tags = get_multi_stage_outputs(
                        cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                        cfg.TEST.PROJECT2IMAGE,base_size
                    )

                    final_heatmaps, tags_list = aggregate_results(
                        cfg, s, final_heatmaps, tags_list, heatmaps, tags
                    )

                final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                tags = torch.cat(tags_list, dim=4)
                group_begin = time.time()
                grouped, scores = parser.parse(
                    final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )
                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )

            if i % cfg.PRINT_FREQ == 0:
                print("finish images: {}".format(i))
                save_img_frame = True
            else:
                save_img_frame = False
            
            #print(final_results)            

            filtered_results = []
            for i in range(len(final_results)):
                filtered_results.append(final_results[i][final_results[i][:,2] > args.score_thr])
            #print(filtered_results)
            
            img_output_dir = os.path.join(final_output_dir, 'result_valid')
            os.makedirs(img_output_dir, exist_ok=True)
            prefix = os.path.join(img_output_dir, str(i))
            vis_frame = save_valid_image(image, filtered_results, '{}.jpg'.format(prefix), dataset='COCO', save=save_img_frame)

            if save_out_video:
                videoWriter.write(vis_frame)

        if save_out_video:
            videoWriter.release()

if __name__ == '__main__':
    main()
