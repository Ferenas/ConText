import os
import glob
from PIL import Image
import numpy as np

import torch, torchvision
from torch.utils.data import Dataset
import torch.distributed as dist
import random 
import os
from os.path import join as osp
import torch.nn.functional as F

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])



def create_grid_from_images(support_img, support_mask, query_img, query_mask, padding,flip = False,reverse=False):
    if reverse:
        support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
    canvas = np.ones((3 * support_img.shape[0] + 2 * padding, 2 * support_img.shape[1] + 2 * padding,support_img.shape[2]))
    canvas[:support_img.shape[0], :support_img.shape[1],:] = support_img
    if flip:
        canvas[:support_img.shape[0], -support_img.shape[1]:,:] = query_img
        canvas[-query_img.shape[0]:, -support_img.shape[1]:,:] = query_mask
        canvas[-query_img.shape[0]:, :query_img.shape[1],:] = support_mask
    else:
        sec_end = query_img.shape[0]+padding+query_img.shape[0]
        canvas[query_img.shape[0]+padding:sec_end, :query_img.shape[1],:] = query_img
        canvas[:support_img.shape[0], -support_img.shape[1]:,:] = support_mask
        canvas[-query_img.shape[0]:, :query_img.shape[1],:] = query_img
        canvas[-query_img.shape[0]:, -support_img.shape[1]:,:] = query_mask

    return canvas


class TextDataset(Dataset):

    def __init__(self, img_src_dir, input_size, padding=4,stride=(224,224),ext_list=('*.png', '*.jpg'),slide=False):
        super(TextDataset, self).__init__()
        self.img_src_dir = img_src_dir

        self.img_path_prompt = osp(img_src_dir,'Images/Train')
        self.ann_path_prompt = osp(img_src_dir,'annotation/Train')
        self.padding = padding
        self.img_path_v = osp(img_src_dir,'Images/Test')
        self.ann_path_v = osp(img_src_dir,'annotation/Test')


        self.eimg_path_prompt = '/mnt/nas/users/dikai.zf/data/TRCG/TRCG_data/data/TotalText_train/images'
        self.eimg_path_prompt = None
        
        if self.eimg_path_prompt != None:
            self.img_meta_prompt = self.build_img_meta(self.eimg_path_prompt)
            self.img_meta_v = self.build_img_meta(self.eimg_path_prompt)
        else:
            self.img_meta_prompt = self.build_img_meta(self.img_path_prompt)
            self.img_meta_v = self.build_img_meta(self.img_path_v)

        self.input_size = input_size
        self.stride = stride
        self.slide = slide

    
    def build_img_meta(self,img_path_dir):
        img_meta = []
        for img in os.listdir(img_path_dir):
            img_meta.append(img)
        return img_meta

    def episode_sampler(self):
        index = random.randint(0, len(self.img_meta_prompt)-1)
        img_path_p = osp(self.img_path_prompt, self.img_meta_prompt[index])
        ann_path_p = osp(self.ann_path_prompt, self.img_meta_prompt[index])
        if self.eimg_path_prompt == None:
            oimg_path_p = None
        else:
            oimg_path_p = osp(self.eimg_path_prompt, self.img_meta_prompt[index])
        
        return img_path_p, ann_path_p,oimg_path_p

    def __len__(self):
        return len(self.img_meta_v)

    def transform(self,img_path,tgt_path):
        image = Image.open(img_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        ori_size = image.size
        img = np.array(image.resize(self.input_size)) / 255.
        tgt = np.array(tgt.resize(self.input_size,Image.NEAREST)) / 255.

        tgt[tgt>= 0.5] = 1
        tgt[tgt < 0.5] = 0


        return img, tgt, ori_size

    def transform_slide(self,img_path,tgt_path):
        image = Image.open(img_path).convert("RGB")
        target = Image.open(tgt_path).convert("RGB")

        ori_size = image.size

        slide_size = (1024,1024)   #(512,512)
        img = np.array(image.resize(slide_size)) / 255.
        tgt = np.array(target.resize((slide_size),Image.NEAREST)) / 255.

        img_flag = np.array(image.resize(self.input_size)) / 255.
        tgt_flag = np.array(target.resize(self.input_size,Image.NEAREST)) / 255.

        tgt[tgt>= 0.5] = 1
        tgt[tgt < 0.5] = 0

        
        return img, tgt, img_flag,tgt_flag,ori_size

    def create_grid(self,img1,img2,padding=1,flip=False,reverse=False):
        h,w,_ = img1.shape
        presented_result = torch.stack([torch.from_numpy(img1.transpose(2,0,1)),torch.from_numpy(img2.transpose(2,0,1))],dim=0)
        final_grid = torchvision.utils.make_grid(presented_result, nrow=1, padding=padding)
        
        C, H, W = final_grid.shape
        final_grid = final_grid[:, padding:H-padding, padding:W-padding]

        final_grid = F.interpolate(final_grid.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True).squeeze(0)        

        return final_grid.permute(1,2,0).numpy()

    def __getitem__(self, index):
        if self.slide:
            return self.__getitem__slide(index)
        else:
            return self.__getitem__normal(index)

    
    def __getitem__normal(self, index):
        img_path_v = osp(self.img_path_v, self.img_meta_v[index])
        ann_path_v = osp(self.ann_path_v, self.img_meta_v[index])
        if self.eimg_path_prompt != None:
            oimg_path_v = osp(self.eimg_path_prompt, self.img_meta_v[index])

        img_path_s,ann_path_s, oimg_path_s = self.episode_sampler()

        img_v, tgt_v, size_org_v = self.transform(img_path_v, ann_path_v)
        img_s, tgt_s, size_org_s = self.transform(img_path_s, ann_path_s)
        
        if oimg_path_s != None:
            oimg_s, _, _ = self.transform(oimg_path_s, ann_path_v)
            oimg_v, _, _ = self.transform(oimg_path_v, ann_path_v)
            c,h,w = img_s.shape
            img_s = self.create_grid(img_s,oimg_s,padding=self.padding)
            img_v = self.create_grid(img_v,oimg_v,padding=self.padding)

        flags = create_grid_from_images(img_s, tgt_s, img_v, tgt_v,self.padding)*255

        ori_tgt_v = np.array(Image.open(ann_path_v).convert("RGB"))
        ori_tgt_v[ori_tgt_v >= 255/2] = 255
        ori_tgt_v[ori_tgt_v < 255/2] = 0


        tgt = np.concatenate((tgt_s, tgt_v), axis=0)
        img = np.concatenate((img_s, img_v), axis=0)


        img = (img - imagenet_mean) / imagenet_std
        tgt = (tgt - imagenet_mean) / imagenet_std

        return {"img":img,"tgt":tgt,"size_org_v":size_org_v,
        "img_name":self.img_meta_v[index],"ori_img_t":ori_tgt_v,
        "flag":flags}


    def __getitem__slide(self, index):
        img_path_v = osp(self.img_path_v, self.img_meta_v[index])
        ann_path_v = osp(self.ann_path_v, self.img_meta_v[index])

        img_path_s,ann_path_s = self.episode_sampler()

        img_s, tgt_s, img_sf,tgt_sf,size_org_s = self.transform_slide(img_path_s, ann_path_s)

        img_v, tgt_v,size_org_v = self.transform(img_path_v, ann_path_v)

        ori_tgt_v = np.array(Image.open(ann_path_v).convert("RGB"))

        flags = create_grid_from_images(img_sf, tgt_sf, img_v, tgt_v,self.padding)*255

        
        

        return {"img":[img_s,img_v],"tgt":[tgt_s,tgt_v],"size_org_v":size_org_v,
        "img_name":self.img_meta_v[index],"ori_img_t":ori_tgt_v,
        "flag":flags}

    def _slide_inference(self,img_v,tgt_v,img_s,tgt_s):
        f_img = []
        f_tgt = []
        crop_ind = []
        h_crop, w_crop = self.input_size
        h_stride, w_stride = self.stride
        h_img, w_img = img_v.shape[:-1]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for i in range(h_grids):
            for j in range(w_grids):
                
                y1 = i * h_stride
                x1 = j * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop,0)
                x1 = max(x2 - w_crop,0)
                crop_img,crop_mask = img_v[y1:y2, x1:x2, :],tgt_v[y1:y2, x1:x2, :]
                img = np.concatenate((img_s, crop_img), axis=0)
                tgt = np.concatenate((tgt_s, crop_mask), axis=0)
                img = (img - imagenet_mean) / imagenet_std
                tgt = (tgt - imagenet_mean) / imagenet_std
                f_img.append(img)
                f_tgt.append(tgt)
                crop_ind.append((y1, y2, x1, x2))

        return np.array(f_img),np.array(f_tgt), np.array(crop_ind)

