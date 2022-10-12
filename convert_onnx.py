import torch.onnx 

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

    input_size = [1, 3, 256, 192]
    Convert_ONNX(model, input_size) 


#Function to Convert to ONNX 
def Convert_ONNX(model, input_size): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "search-x.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == '__main__':
    main()