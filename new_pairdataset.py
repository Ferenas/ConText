# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random
import torch.nn.functional as F

from PIL import Image
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform
import torchvision
import scipy.ndimage
class PairDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        json_path_root: str,
        transform: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        transform3: Optional[Callable] = None,
        transform_seccrop: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        masked_position_generator: Optional[Callable] = None,
        use_two_pairs: bool = True,
        half_mask_ratio:float = 0.,
        val_mode: bool = False,
        data: list = ['totaltext'],
        upper_perform = False,
        prompt = False,
        shot = 1,
        demo=False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.pairs = []
        self.demo_pairs = []
        self.weights = []
        self.data = data
        self.upper_perform = upper_perform
        #Based on the data length to assign a weight

        # type_weight_list = [1.0, 0.2, 0.15, 0.25, 0.2, 0.15, 0.05, 0.05]
        json_path_list = os.listdir(json_path_root)
        type_weight_list = [1.0]*len(json_path_list)
        self.val_mode = val_mode
        self.prompt = prompt
        self.shot = shot
        self.demo = demo
        


        
    
        for idx, json_path in enumerate(json_path_list):
            # json_path.split('_')[0]
            if  ('_').join(json_path.split('_')[:2]) not in data:
                continue
            
            # else:
            json_path = os.path.join(json_path_root,json_path)

            if demo:
                cur_pairs = json.load(open(json_path))['prompt']
                pro_pairs = json.load(open(json_path))['demo']
                self.demo_pairs.extend(pro_pairs)
            else:
                cur_pairs = json.load(open(json_path))
            
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num]*cur_num)
            print(json_path, type_weight_list[idx])

        self.use_two_pairs = use_two_pairs
        if self.use_two_pairs:
            self.pair_type_dict = {}
            for idx, pair in enumerate(self.pairs):
                if "type" in pair:
                    if pair["type"] not in self.pair_type_dict:
                        self.pair_type_dict[pair["type"]] = [idx]
                    else:
                        self.pair_type_dict[pair["type"]].append(idx)
            for t in self.pair_type_dict:
                print(t, len(self.pair_type_dict[t]))


        self.transforms = ThreeStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.transforms_seccrop = PairStandardTransform(transform_seccrop, target_transform) if transform_seccrop is not None else None
        self.masked_position_generator = masked_position_generator
        self.half_mask_ratio = half_mask_ratio

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                if os.path.isfile(os.path.join(self.root, path)):
                    img = Image.open(os.path.join(self.root, path))
                else:
                    img = Image.open(path)
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def _combine_images(self, image, image2, interpolation='bicubic'):
        # image under image2
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        pair = self.pairs[index]
        if self.demo:
            pair = self.demo_pairs[index]
        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])
        if 'imagere_path' in pair.keys():
            image_ori = self._load_image(pair['imagere_path'])
        else:
            image_ori = image
        
        ori_size = image.size
        image_ori = image_ori.resize(ori_size, Image.LANCZOS)

        img_name = pair['image_path'].split('/')[-1]

        # decide mode for interpolation
        pair_type = pair['type']
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
            
        # no aug for instance segmentation
        if "inst" in pair['type'] and self.transforms2 is not None:
            cur_transforms = self.transforms2
        elif "pose" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms

        def grounded(input):
            mid = 255 /2
            input[input>=mid] = 255
            input[input<mid] = 0
            return input
        

        if "textseg" not in self.data[0]:
            ori_mask_with_size = grounded(np.array(target))
        else:
            ori_mask_with_size = np.array(target)

        ori_imgre_with_size = np.array(image_ori)
        image, target,image_ori = cur_transforms(image, target, image_ori,interpolation1, interpolation2)

        imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
        imagenet_std=torch.tensor([0.229, 0.224, 0.225])

        # if self.prompt:
            # target_01 = torch.round(target*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))[0]
            # label_array, _ = scipy.ndimage.label(target_01.numpy())
            # character_id = np.unique(label_array)[1:]
            # new_mask,comple_new_mask = np.zeros_like(label_array),np.zeros_like(label_array)
            # erase_prob = 0.5
            # select = np.random.rand(*character_id.shape) < erase_prob
            # selected_id,verse = character_id[select],character_id[~select]
            
            # new_mask[np.isin(label_array, selected_id)] = 1
            # comple_new_mask[np.isin(label_array, verse)] = 1

            # comple_new_mask = torch.from_numpy(comple_new_mask).unsqueeze(0).repeat(3,1,1)
            # new_mask = torch.from_numpy(new_mask).unsqueeze(0).repeat(3,1,1)

            # # target = (new_mask)# - imagenet_mean.reshape(3,1,1)) / imagenet_std.reshape(3,1,1)
            
            # substitute_mark=torch.tensor([0,255,0]) // 255
            # new_mask_highlight = (new_mask*substitute_mark.reshape(3,1,1) - imagenet_mean.reshape(3,1,1)) / imagenet_std.reshape(3,1,1)
            # target = (new_mask  - imagenet_mean.reshape(3,1,1)) / imagenet_std.reshape(3,1,1)
            # new_image = image*(1-new_mask) + new_mask_highlight*new_mask
            # image_ori = image*(1-new_mask) + image_ori*new_mask
            # image = new_image

            # ori_mask_with_size = F.interpolate(new_mask.unsqueeze(0).float()*255,ori_mask_with_size.shape[:2],mode='bilinear')[0].permute(1,2,0).numpy().astype(np.uint8)


        if self.shot > 1:
            image_batch,image_ori_batch, target_batch = [],[],[]
            for _ in range(self.shot):
                pair_type = pair['type']
                pair2_index = random.choice(self.pair_type_dict[pair_type])
                pair2 = self.pairs[pair2_index]
                image2 = self._load_image(pair2['image_path'])
                target2 = self._load_image(pair2['target_path'])
                if 'imagere_path' in pair2.keys():
                    image2_ori = self._load_image(pair2['imagere_path'])
                else:
                    image2_ori = image2

                image2_ori = image2_ori.resize(image2.size, Image.LANCZOS)
                # assert pair2['type'] == pair_type
                
                image2, target2, image2_ori = cur_transforms(image2, target2, image2_ori, interpolation1, interpolation2)

                if image2.shape != image.shape:
                    image2 = torch.squeeze(F.interpolate(torch.unsqueeze(image2, 0), image.shape[1:], mode='bilinear'), dim=0)
                    target2 = torch.squeeze(F.interpolate(torch.unsqueeze(target2, 0), image.shape[1:], mode='nearest'), dim=0)
                    image2_ori = torch.squeeze(F.interpolate(torch.unsqueeze(image2_ori, 0), image.shape[1:], mode='bilinear'), dim=0)
                
                flag,flag_v = self.new_create_grid([image2,image],[target2,target],[image2_ori, image_ori])
                flag = torch.round((flag*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))*255)
                flag_v = torch.round((flag_v*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))*255)
                
                if image2.shape == image.shape:
                    image_new = self._combine_images(image2, image, interpolation1)
                    target_new = self._combine_images(target2, target, interpolation2)
                    image_ori_new = self._combine_images(image2_ori, image_ori, interpolation1)
                
                image_batch.append(image_new)
                image_ori_batch.append(image_ori_new)
                target_batch.append(target_new)

            image = np.stack(image_batch, axis=0)
            target = np.stack(target_batch, axis=0)
            image_ori = np.stack(image_ori_batch, axis=0)


        else:
            if self.use_two_pairs:
                pair_type = pair['type']

                # if self.val_mode:
                #     self.pairs_val = []
                #     json_path_root = 'data/Train'
                #     json_path_list = os.listdir(json_path_root)
                #     data_val = ['TotalText_Train']  #'HierText_Train','HierText_Train',
                #     for idx, json_path in enumerate(json_path_list):
                #     # json_path.split('_')[0]
                #         if  ('_').join(json_path.split('_')[:2]) not in data_val:
                #             continue
                    
                #     # else:
                #         json_path = os.path.join(json_path_root,json_path)
                #         cur_pairs = json.load(open(json_path))
                #         self.pairs_val.extend(cur_pairs)
                #         cur_num = len(cur_pairs)


                #     self.pair_type_dict_val = {}
                #     for idx, pair in enumerate(self.pairs_val):
                #         if "type" in pair:
                #             if pair["type"] not in self.pair_type_dict_val:
                #                 self.pair_type_dict_val[pair["type"]] = [idx]
                #             else:
                #                 self.pair_type_dict_val[pair["type"]].append(idx)
                #     # for t in self.pair_type_dict_val:
                #     #     print("Select from:",t, len(self.pair_type_dict_val[t]))

                #     pair2_index = random.choice(self.pair_type_dict_val[pair["type"]])
                #     pair2 = self.pairs_val[pair2_index]
                
                # # sample the second pair belonging to the same type
                # # index = random.randint(0, len(self.img_meta_prompt)-1)
                # # pair2_index = random.randint(0, len(self.pair_type_dict_train[pair_type])-1)
                # else:
                #     pair2_index = random.choice(self.pair_type_dict[pair_type])
                #     pair2 = self.pairs[pair2_index]
                


                if self.val_mode and self.upper_perform:
                    pair2 = self.pairs[index]
                elif self.prompt:
                    pair2_index = random.choice(self.pair_type_dict[pair_type])
                    while pair2_index == index:
                        pair2_index = random.choice(self.pair_type_dict[pair_type])
                    pair2 = self.pairs[pair2_index]

                elif self.val_mode:                
                    pair2_index = random.choice(self.pair_type_dict[pair_type])
                    pair2 = self.pairs[pair2_index]
                else:
                    pair2_index = random.choice(self.pair_type_dict[pair_type])
                    pair2 = self.pairs[pair2_index]
                    if torch.rand(1)[0] < 0.25:
                        pair2 = self.pairs[index]
                    # else:
                    #     pair2_index = random.choice(self.pair_type_dict[pair_type])
                    #     pair2 = self.pairs[pair2_index]
                
                
                image2 = self._load_image(pair2['image_path'])
                target2 = self._load_image(pair2['target_path'])
                if 'imagere_path' in pair2.keys():
                    image2_ori = self._load_image(pair2['imagere_path'])
                else:
                    image2_ori = image2

                image2_ori = image2_ori.resize(image2.size, Image.LANCZOS)
                # assert pair2['type'] == pair_type
                
                image2, target2, image2_ori = cur_transforms(image2, target2, image2_ori, interpolation1, interpolation2)
                
                # if self.prompt:
                #     if self.upper_perform:
                #         new_mask = comple_new_mask
                #     else:
                #         target_01 = torch.round(target2*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))[0]
                #         label_array, _ = scipy.ndimage.label(target_01.numpy())
                #         character_id = np.unique(label_array)[1:]
                #         new_mask,comple_new_mask = np.zeros_like(label_array),np.zeros_like(label_array)
                #         erase_prob = 0.5
                #         select = np.random.rand(*character_id.shape) < erase_prob
                #         selected_id,verse = character_id[select],character_id[~select]
                #         new_mask[np.isin(label_array, selected_id)] = 1
                #         comple_new_mask[np.isin(label_array, verse)] = 1
                #         comple_new_mask = torch.from_numpy(comple_new_mask).unsqueeze(0).repeat(3,1,1)
                #         new_mask = torch.from_numpy(new_mask).unsqueeze(0).repeat(3,1,1)
                    
                #     substitute_mark=torch.tensor([255,0,0]) // 255

                #     new_mask_highlight = (new_mask*substitute_mark.reshape(3,1,1) - imagenet_mean.reshape(3,1,1)) / imagenet_std.reshape(3,1,1)
                #     target2 = (new_mask  - imagenet_mean.reshape(3,1,1)) / imagenet_std.reshape(3,1,1)
                #     new_image2 = image2*(1-new_mask) + new_mask_highlight*new_mask
                #     image2_ori = image2*(1-new_mask) + image2_ori*new_mask
                #     image2 = new_image2
                    


                # if self.val_mode:
                #     image2, target2, image2_ori = cur_transforms(image2, target2, image2_ori, interpolation1, interpolation2)

                    # word_target = self._load_image(pair2['word_path'])
                    # target2_01 = torch.round(target2*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))[0]
                    # random_target2_01 = torch.rand(target2_01.shape)
                    # erase_prob = 0.8 
                    # erase_mask = (random_target2_01 < erase_prob) & (target2_01 == 1)
                    # target2_01[erase_mask] = 0
                    # target2_01 = target2_01.unsqueeze(0)
                    # image2_new = image2*target2_01 + image2_ori*(1-target2_01)
                    # image2_ori_new = image2*(1-target2_01) + image2_ori*target2_01
                    # target2 = (target2_01.repeat(3,1,1)-imagenet_mean.reshape(3,1,1)) / imagenet_std.reshape(3,1,1)

                
                if image2.shape != image.shape:
                    image2 = torch.squeeze(F.interpolate(torch.unsqueeze(image2, 0), image.shape[1:], mode='bilinear'), dim=0)
                    target2 = torch.squeeze(F.interpolate(torch.unsqueeze(target2, 0), image.shape[1:], mode='bilnear'), dim=0)
                    image2_ori = torch.squeeze(F.interpolate(torch.unsqueeze(image2_ori, 0), image.shape[1:], mode='bilinear'), dim=0)
                
                flag,flag_v = self.new_create_grid([image2,image],[target2,target],[image2_ori, image_ori])
                flag = torch.round((flag*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))*255)
                flag_v = torch.round((flag_v*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))*255)
                
                if image2.shape == image.shape:
                    image = self._combine_images(image2, image, interpolation1)
                    target = self._combine_images(target2, target, interpolation2)
                    image_ori = self._combine_images(image2_ori, image_ori, interpolation1)
                else:
                    image = [image2, image]
                    target = [target2, target]
                    image_ori = [image2_ori, image_ori]



        
        if self.val_mode:
            return [image, target, image_ori],ori_size,flag,flag_v,[ori_mask_with_size,ori_imgre_with_size],img_name
        



        use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
        if (self.transforms_seccrop is None) or ("inst" in pair['type']) or ("pose" in pair['type']) or use_half_mask:
            pass
        else:
            image, target = self.transforms_seccrop(image, target, interpolation1, interpolation2)
        
        valid = torch.ones_like(target)
        _,h,w = target.shape
        # ori_mask = F.interpolate(torch.from_numpy(ori_mask_with_size).permute(2,0,1).unsqueeze(0),(h,w),mode='nearest')[0]
        
        # image = torch.round((image*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))*255).permute(1,2,0)
        # Image.fromarray(image.numpy().astype(np.uint8)).save("erasing.png")

        # image_ori = torch.round((image_ori*imagenet_std.reshape(3,1,1)+imagenet_mean.reshape(3,1,1))*255).permute(1,2,0)
        # Image.fromarray(image_ori.numpy().astype(np.uint8)).save("aerasing.png")
        # ori_mask = F.interpolate(torch.from_numpy(ori_mask_with_size).permute(2,0,1).unsqueeze(0),(h,w),mode='nearest')[0]

        assert image.shape == target.shape

        # attentive_mask = self.mask_to_patches(ori_mask/255.).numpy().astype(np.int32)
        
        if use_half_mask:
            num_patches = self.masked_position_generator.num_patches
            mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
            mask[mask.shape[0]//2:, :] = 1
        else:
            mask = self.masked_position_generator()
            # mask[:mask.shape[0]//2, :] = 0

        # if torch.rand(1)[0] < 0.5:
        #     mask_final = attentive_mask
        # else:
        #     mask_final = mask
        
        # import torch.nn.functional as F
        # z = F.interpolate(attentive_mask.unsqueeze(0).unsqueeze(0),(h,w),mode='nearest')[0,0]
        # z[z==1.] = 255
        # Image.fromarray(z.numpy().astype(np.uint8)).save("aerasing.png")

        # z2 = F.interpolate(torch.from_numpy(mask).to(torch.float32).unsqueeze(0).unsqueeze(0),(h,w),mode='nearest')[0,0]
        # z2[z2==1.] = 255
        # Image.fromarray(z2.numpy().astype(np.uint8)).save("erasing.png")

        # ori_mask[ori_mask==1.] = 255
        # Image.fromarray(ori_mask.permute(1,2,0).numpy().astype(np.uint8)).save("ori_mask.png")

        # if "nyuv2_image2depth" in pair_type:
        #     thres = torch.ones(3) * (1e-3 * 0.1)
        #     thres = (thres - imagenet_mean) / imagenet_std
        #     valid[target < thres[:, None, None]] = 0
        # elif "ade20k_image2semantic" in pair_type:  #totaltext_image2semantic
        #     thres = torch.ones(3) * (1e-5) # ignore black
        #     thres = (thres - imagenet_mean) / imagenet_std
        #     valid[target < thres[:, None, None]] = 0
        # elif "coco_image2panoptic_sem_seg" in pair_type:
        #     thres = torch.ones(3) * (1e-5) # ignore black
        #     thres = (thres - imagenet_mean) / imagenet_std
        #     valid[target < thres[:, None, None]] = 0
        # elif "image2pose" in pair_type:
        #     thres = torch.ones(3) * (1e-5) # ignore black
        #     thres = (thres - imagenet_mean) / imagenet_std
        #     valid[target > thres[:, None, None]] = 10.0
        #     fg = target > thres[:, None, None]
        #     if fg.sum() < 100*3:
        #         valid = valid * 0.
        # elif "image2panoptic_inst" in pair_type:
        #     thres = torch.ones(3) * (1e-5) # ignore black
        #     thres = (thres - imagenet_mean) / imagenet_std
        #     fg = target > thres[:, None, None]
        #     if fg.sum() < 100*3:
        #         valid = valid * 0.



        
        return image, target, image_ori, mask, valid


    def new_create_grid(self,c_img,c_mask,c_imgo,padding=2):
        _,h,w = c_img[0].shape



        c_img = torch.cat([c.unsqueeze(0) for c in c_img],dim=0)
        c_mask = torch.cat([c.unsqueeze(0) for c in c_mask],dim=0)
        c_imgo = torch.cat([c.unsqueeze(0) for c in c_imgo],dim=0)
        
        final_image_vertical = torch.cat([c_img,c_img[-1:],c_imgo,c_imgo[-1:],c_mask,c_mask[-1:]],dim=0)
        final_image = torch.cat([final_image_vertical[::3],final_image_vertical[1::3],final_image_vertical[2::3]],dim=0)
        # final_image_vertical = torch.cat([c_img[0:],c_imgo[0:],c_imgo],dim=0)

        # final_image_vertical = torch.cat([c_img[-1:],c_img[-1:],c_imgo[-1:],c_imgo[-1:],c_mask[-1:],c_mask[-1:]],dim=0)  #vertical
        # final_image_vertical = torch.cat([c_img[-1:],c_imgo[-1:],c_mask[-1:],c_img[-1:],c_imgo[-1:],c_mask[-1:]],dim=0)  #normal

        length = final_image.shape[0] // 3 
        
        mask_sign = torch.ones_like(final_image)
        mask_sign[-1:,:,:,:] = 0
        final_grid = torchvision.utils.make_grid(final_image, nrow=length, padding=padding, pad_value=1.0)
        final_grid_vertical = torchvision.utils.make_grid(final_image, nrow=length, padding=padding, pad_value=1.0)
        # final_grid_vertical = torchvision.utils.make_grid(final_image_vertical, nrow=c_img.shape[0], padding=padding, pad_value=1.0)
        # final_grid_vertical = torchvision.utils.make_grid(final_image_vertical, nrow=c_img.shape[0]+1, padding=padding, pad_value=1.0)

        final_sign = torchvision.utils.make_grid(mask_sign, nrow=length, padding=padding, pad_value=1.0)
        
        C, H, W = final_grid.shape
        final_grid = final_grid[:, padding:H-padding, padding:W-padding]
        final_sign = final_sign[:, padding:H-padding, padding:W-padding]
        final_grid_vertical = final_grid_vertical[:, padding:H-padding, padding:W-padding]
        # final_grid = F.interpolate(final_grid.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True)[0]
        # final_sign = F.interpolate(final_sign.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True)[0]
        return final_grid,final_grid_vertical

    def mask_to_patches(self,mask, patch_size=16):
        # assert mask.shape == (self.img_size, self.img_size), "Input mask must have shape (224, 224)"
        pooled_mask = F.avg_pool2d(mask[:1,:,:], kernel_size=patch_size, stride=patch_size)
        valid_element = 0.1
        pooled_mask = (pooled_mask > valid_element).float()
        return pooled_mask[0]

    def __len__(self) -> int:
        if self.demo:
            return len(self.demo_pairs)
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target

class ThreeStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, input2, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target, input2 = self.transform(input, target,input2, interpolation1, interpolation2)
        return input, target, input2