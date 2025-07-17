import os
import argparse

import torch
import numpy as np
import torch.multiprocessing as mp
from context_engine import *
import models_context
from util.segeval_utils import seed_everything
from thop import profile

def get_args_parser():
    
    parser = argparse.ArgumentParser('ConText inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')   
    parser.add_argument('--img_path', type=str, help='path to img',default='')
    parser.add_argument('--deepspeed',type=str,default='true')
    parser.add_argument('--input_size',type=int,default=1280)
    parser.add_argument('--model', type=str, help='context attribute',
                        default='seggpt_vit_large_patch32_input2048x1024')  
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default='examples/hmbb_2.jpg',)
    parser.add_argument('--shot', type=int, help='path to input video to be tested',
                        default=1)
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./all_output')
    parser.add_argument('--upper_per', action='store_true',
                        default=False)
    parser.add_argument('--slide', action='store_true',
                        default=False)
    parser.add_argument('--data_used', type=str, nargs='+',help='data used for evaluation', default=['textseg'])
    
    return parser.parse_args()


def prepare_model(chkpt_dir, arch):
    model = getattr(models_context, arch)()
    model.seg_type = 'semantic'
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['module'], strict=False)
    model.eval()
    return model 



if __name__ == '__main__':


    args = get_args_parser()
    slide = args.slide
    seed_everything(0)
    
    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model)
    print('Model loaded.')

    os.makedirs(args.output_dir, exist_ok=True)
    multi_model_inference_batch_image(model, device, args.img_path, args.output_dir,args.data_used,args.ckpt_path, \
        slide,args.upper_per,args.input_size)
    

    print('Finished.')
