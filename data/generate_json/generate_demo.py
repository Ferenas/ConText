import os
import glob
import json
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import shutil

def get_args_parser():
    parser = argparse.ArgumentParser('TotalText segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['training', 'validation','Train','Test'], default='Train')
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='data')

    return parser.parse_args()



if __name__ == '__main__':
    
    ####Mask Regenerate for hierText

    dataset_name = 'Demo'

    args = get_args_parser()

    root_data_dir = "demo_data"

    os.makedirs(args.output_dir+'/Test', exist_ok=True)
    save_test_path = os.path.join(args.output_dir+'/Test', f"Demo_Test_image_semantic_10.json")
    val_data_list = os.listdir(os.path.join(root_data_dir, 'ori'))
    output_dict = []

    for idx,image_name in tqdm(enumerate(val_data_list)):
        pair_dict = {}

        pair_dict["image_path"] = os.path.join("ori", image_name) #ori_img
        pair_dict["target_path"] = "seg/" + image_name[:-4]+'.png'  #seg mask
        pair_dict["imagere_path"] = os.path.join("rem", image_name) #removal_img
        pair_dict["type"] = f"Demo_image2semantic"
        
        output_dict.append(pair_dict)
    
    json.dump(output_dict, open(save_test_path, 'w'),indent=4)
    print(f"Finish Test data json {len(output_dict)}")








    







        
    

    
