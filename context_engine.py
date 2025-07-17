import torch
import torch.nn.functional as F
import numpy as np
import cv2
from util.data_utils import TextDataset,create_grid_from_images

from PIL import Image
import os
from util.segeval_utils import _calc_metric, _calc_metric_generation, SegmentationEvaluator
from data.pairdataset import PairDataset
from torch.utils.data import Subset
import data.pair_transforms as pair_transforms
import data.three_transforms as three_transforms
from thop import profile
import time 


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Cache(list):
    def __init__(self, max_size=0):
        super().__init__()
        self.max_size = max_size

    def append(self, x):
        if self.max_size <= 0:
            return
        super().append(x)
        if len(self) > self.max_size:
            self.pop(0)



def gaussian_kernel_2d(size, sigma):

    x = torch.arange(size).float() - (size - 1) / 2
    y = torch.arange(size).float() - (size - 1) / 2
    x, y = torch.meshgrid(x, y)  
    kernel = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)  
    kernel = kernel / kernel.sum()  
    return kernel

def gaussian_blur(input_tensor, kernel_size=5, sigma=1.0):

    kernel = gaussian_kernel_2d(kernel_size, sigma).unsqueeze(0).unsqueeze(0)  
    nkernel = kernel.repeat(3, 1, 1, 1)  
    input_tensor = input_tensor.unsqueeze(0).permute(0,3,1,2)

    smoothed_tensor = F.conv2d(input_tensor, nkernel, padding=kernel_size//2, groups=input_tensor.size(1))
    
    return smoothed_tensor.permute(0,2,3,1).squeeze(0)  


def grounded_img(input):
    dtype = input.dtype
    black = torch.tensor([0, 0, 0], dtype=dtype)
    white = torch.tensor([255, 255, 255], dtype=dtype)


    dist_to_black = torch.norm(input - black, dim=2)
    dist_to_white = torch.norm(input - white, dim=2)

    mask = dist_to_black <= dist_to_white

    input[mask] = black
    input[~mask] = white
    return input

@torch.no_grad()
def run_one_image(img, tgt, img_ori, model, device):
    model = model.to(device)

    # make it a batch-like
    x = torch.einsum('nhwc->nchw', img)

    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    # make it a batch-like
    img_ori = torch.einsum('nhwc->nchw', img_ori)


    #Modify to adaptable patch_size
    b,_,h,w = x.shape
    new_num_patches = h*w//(model.patch_size*model.patch_size)
    bool_masked_pos = torch.zeros(new_num_patches)
    bool_masked_pos[new_num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    
    valid = torch.ones_like(tgt)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if x.shape[0] > 1 else -1
    y, mask = model([x.float().to(device),img_ori.float().to(device)], tgt.float().to(device), 
    bool_masked_pos.to(device),bool_masked_pos.to(device),valid.float().to(device), seg_type.to(device), feat_ensemble,val=True) 
    y = torch.einsum('nchw->nhwc', y)


    output = y[:1].max(dim=0)[0][y.shape[1]//2:, :, :].detach().cpu()  #mask
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output = grounded_img(output) / 255

    ###Here if you want to get strict erasing evaluation results, using ground-truth mask for removal
    # tgt_de = model.denorm(tgt)
    # tgt_01 = tgt_de[:,y.shape[1]//2:,:].permute(1,2,0).detach().cpu()
    # output_re = y[1][y.shape[1]//2:, :, :].detach().cpu()*(tgt_01) + (1-tgt_01)*img_ori[0,:,y.shape[1]//2:, :].permute(1,2,0).detach().cpu()
    
    ####Normally, You can use the generated segmentation mask
    dilated_mask = output[:,:,0].unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1,1,7,7),dtype=torch.float64) #Gaussian bluring for better removal visual effect, you can ajust this parameter.
    for i in range(1):
        dilated_mask = F.conv2d(dilated_mask.to(torch.float64), kernel, padding=3)
    dilated_mask = (dilated_mask > 0).float()  # 转换回二进制掩膜
    mask_gen = dilated_mask.squeeze(0).squeeze(0).unsqueeze(-1).repeat(1,1,3)  #.squeeze(0).squeeze(0)
    output_re = y[1][y.shape[1]//2:, :, :].detach().cpu()*(mask_gen) + (1-mask_gen)*(img_ori[0,:,y.shape[1]//2:, :].permute(1,2,0).detach().cpu())



    output = y[:1].max(dim=0)[0][y.shape[1]//2:, :, :].detach().cpu()  #mask
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output_re = torch.clip((output_re * imagenet_std + imagenet_mean) * 255, 0, 255)

    return output,output_re



def infer_and_evaluate(device,model,data_loader, \
    input_size,padding,return_dict,slide,crop_size,stride,re_out_dir,draw_out_dir,flag_out_dir):
        all_iou = 0
        all_mse = 0
        eval_re = np.array([0.,0.,0.,0.,0.,0.])
        Eval = SegmentationEvaluator(2)
        for idx,(batch_list, size,flag,flag_v,ori,img_name) in enumerate(data_loader):
            img,tgt,img_ori = batch_list[0],batch_list[1],batch_list[2]
            ori_mask,ori_imgre = ori[0],ori[1]

            if isinstance(img,list):
                img_s,img_v = img[0].to(device).permute(0,2,3,1),img[1].to(device).permute(0,2,3,1)
                tgt_s,tgt_v = tgt[0].to(device).permute(0,2,3,1), tgt[1].to(device).permute(0,2,3,1)
                img_s_ori,img_v_ori = img_ori[0].to(device).permute(0,2,3,1), img_ori[1].to(device).permute(0,2,3,1)
            else:
                if len(img.shape) > 4:
                    img,tgt,img_ori = img[0],tgt[0],img_ori[0]
                img, tgt = img.to(device).permute(0,2,3,1), tgt.to(device).permute(0,2,3,1)
                img_ori = img_ori.to(device).permute(0,2,3,1)
                img_s,img_v = img[:,:input_size[0]], img[:,input_size[0]:]
                tgt_s,tgt_v = tgt[:,:input_size[0]], tgt[:,input_size[0]:]
                img_s_ori,img_v_ori = img_ori[:,:input_size[0]], img_ori[:,input_size[0]:]



            if slide:
                b,h_img, w_img,_ = img_v.shape
                h_crop, w_crop = crop_size
                h_stride, w_stride = stride

                slide_img_s = F.interpolate(img_s.permute(0,3,1,2),crop_size,mode='bilinear').permute(0,2,3,1)
                slide_tgt_s = F.interpolate(tgt_s.permute(0,3,1,2),crop_size,mode='bilinear').permute(0,2,3,1)
                slide_img_s_ori = F.interpolate(img_s_ori.permute(0,3,1,2),crop_size,mode='bilinear').permute(0,2,3,1)
                
                h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
                preds,preds_img = torch.zeros(img_v.shape),torch.zeros(img_v.shape)
                count_mat,count_mat_pred = torch.zeros((b,h_img,w_img,1)),torch.zeros((b,h_img,w_img,1))


                for i in range(h_grids):
                    for j in range(w_grids):
                        
                        y1 = i * h_stride
                        x1 = j * w_stride
                        y2 = min(y1 + h_crop, h_img)
                        x2 = min(x1 + w_crop, w_img)
                        y1 = max(y2 - h_crop,0)
                        x1 = max(x2 - w_crop,0)

                        crop_img_v = img_v[:,y1:y2, x1:x2, :]
                        crop_img_o = img_v_ori[:,y1:y2, x1:x2, :]
                        crop_mask = tgt_v[:,y1:y2, x1:x2, :]

                        img = torch.cat((slide_img_s,crop_img_v), dim=1)
                        tgt = torch.cat((slide_tgt_s, crop_mask), dim=1)
                        img_o = torch.cat((slide_img_s_ori, crop_img_o), dim=1)
                        output,output_re = run_one_image(img, tgt, img_o, model, device,cot)

                        preds += F.pad(output[None, ...].permute(0, 3, 1, 2),
                                    (int(x1), int(preds.shape[2] - x2), int(y1),
                                        int(preds.shape[1] - y2))).permute(0, 2, 3, 1)
                        preds_img += F.pad(output_re[None, ...].permute(0, 3, 1, 2),
                                    (int(x1), int(preds_img.shape[2] - x2), int(y1),
                                        int(preds_img.shape[1] - y2))).permute(0, 2, 3, 1)
                        count_mat[:, y1:y2, x1:x2,:] += 1
                        count_mat_pred[:, y1:y2, x1:x2,:] += 1

                output = (preds / count_mat)[0]
                output_re = (preds_img / count_mat)[0]

            else:
                output,output_re = run_one_image(img, tgt, img_ori, model, device)


            flags = flag[0].permute(1,2,0).numpy()
            flag_size_out = output.shape[:2]
            flags[flag_size_out[0]+padding:flag_size_out[0]*2+padding, -flag_size_out[1]:,:] = grounded_img(output)  #grounded_img
            flags[flag_size_out[0]+padding:flag_size_out[0]*2+padding, flag_size_out[0]+padding:flag_size_out[0]*2+padding,:] = output_re  #grounded_img
            
        
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2), 
                size=[size[1], size[0]], 
                mode='bilinear',  #'bilinear'
            ).permute(0, 2, 3, 1)[0].detach().cpu()

            output_re = torch.round(F.interpolate(
                output_re[None, ...].permute(0, 3, 1, 2), 
                size=[size[1], size[0]], 
                mode='bilinear',
            ).permute(0, 2, 3, 1))[0].detach().cpu().numpy()


            output = grounded_img(output)
            output_re = np.round(output_re)
            
            
            if 'textseg' in data_loader.dataset.dataset.data[0]:
                ignore = 120  #Ignore pixel for textseg
            else:
                ignore = None


            current_metric_generation = _calc_metric_generation(output_re,ori_imgre[0].numpy(),ori_mask[0].numpy())
            current_metric = _calc_metric(output.numpy(),ori_mask[0].numpy())

            Eval.add_batch(ori_mask[0].numpy(),output.numpy(),ignore)
            iou = current_metric['iou']*100
            mse = current_metric_generation[0]
            all_iou += iou
            eval_re += current_metric_generation
            if device == 0:
                print(f'current_fiou_mse / size is {iou:.2f}_{mse} / {size[1].numpy()}*{size[0].numpy()}')


            mask_gray = cv2.cvtColor(output.numpy().astype(np.uint8), cv2.COLOR_BGR2GRAY)
            output_re = cv2.inpaint(output_re.astype(np.uint8), mask_gray, inpaintRadius=25, flags=cv2.INPAINT_TELEA)

            # Save Seg  
            output = Image.fromarray(output.numpy().astype(np.uint8))
            output.save(os.path.join(draw_out_dir,img_name[0]))  #resize((512,512))

            # Save Removal
            output_re = Image.fromarray(output_re.astype(np.uint8))
            output_re.save(os.path.join(re_out_dir,img_name[0]))  #resize((512,512))

            # Save All results, the first line is the in-context demonstration, the second line is the generated answer, the last one is GT for generated answer
            flags = Image.fromarray(flags.astype(np.uint8))
            flags.save(os.path.join(flag_out_dir,f'{iou:.2f}_'+img_name[0]))

        return_dict[device] = {'single': eval_re, 'confusion': Eval.confusion_matrix,'f_iou':all_iou}

        


def multi_model_inference_batch_image(model, device, img_path, out_path,data_used,model_name,slide,\
    upper_per,input_size):

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    process = []
    manager = mp.Manager()
    return_dict = manager.dict()
    num_gpus = torch.cuda.device_count()
    
    vis_out = 'vis'
    draw_out_dir = ('_').join(model_name.split('/')[1:]) + f'_{data_used[0]}_seg_' + f'upper_{upper_per}'
    re_out_dir = ('_').join(model_name.split('/')[1:]) + f'_{data_used[0]}_rem_' + f'upper_{upper_per}'
    flag_out_dir = ('_').join(model_name.split('/')[1:]) + f'_{data_used[0]}_all_' + f'upper_{upper_per}'

    with open(os.path.join(out_path, 'log.txt'), 'w') as log:
        log.write('start_evaluation' + '\n')

    draw_out_dir = os.path.join(vis_out,draw_out_dir)
    if os.path.exists(draw_out_dir):
        if len(os.listdir(draw_out_dir)) > 0:
            os.system(f'rm -rf {draw_out_dir}')
    os.makedirs(draw_out_dir,exist_ok=True)

    re_out_dir = os.path.join(vis_out,re_out_dir)
    if os.path.exists(re_out_dir):
        if len(os.listdir(re_out_dir)) > 0:
            os.system(f'rm -rf {re_out_dir}')
    os.makedirs(re_out_dir,exist_ok=True)

    flag_out_dir = os.path.join(vis_out,flag_out_dir)
    if os.path.exists(flag_out_dir):
        if len(os.listdir(flag_out_dir)) > 0:
            os.system(f'rm -rf {flag_out_dir}')
    os.makedirs(flag_out_dir,exist_ok=True)

    if slide:
        if input_size == 1024:
            input_size = (input_size*2,input_size*2)
            crop_size = (input_size[0],input_size[1])
            stride = (512,512)
        else:
            # You may set these parameters according to your own needs
            input_size = (1536,1536)
            crop_size = (1280,1280)
            stride = (512,512)
    else:
        input_size = crop_size = stride = (input_size,input_size)
    padding = 2

    transform_val = three_transforms.Compose([
            three_transforms.Resize(input_size),  
            three_transforms.ToTensor(),
            three_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = PairDataset(img_path, json_path_root='data/Test', transform=transform_val,
                            val_mode=True,data=data_used,upper_perform=upper_per)
    

    dataset_size,indices_all = len(dataset),list(range(len(dataset)))
    split_indices = np.array_split(indices_all,num_gpus)
    for gpu_id in range(num_gpus):
        subset_data = Subset(dataset, split_indices[gpu_id])
        data_loader = torch.utils.data.DataLoader(
        subset_data,
        batch_size=1,
        shuffle=False,
        num_workers=4)
        p = mp.Process(target=infer_and_evaluate, args = (gpu_id,model,data_loader,input_size,padding,
                                                    return_dict,slide,crop_size,stride,re_out_dir,draw_out_dir,flag_out_dir))
        process.append(p)
        p.start()
    



    for p in process:
        p.join()



    total_res = np.array([0.,0.,0.,0.,0.,0.])
    confusion = 0
    all_iou = 0

    for r in return_dict.values():
        total_res += r['single']
        confusion += r['confusion']
        all_iou += r['f_iou']


    def F_Score(confusion_matrix):
        TP = np.diag(confusion_matrix)
        FP = np.sum(confusion_matrix, axis=0) - TP 
        FN = np.sum(confusion_matrix, axis=1) - TP
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)
        return P, R, F

    mIoU = np.diag(confusion) / (
                    np.sum(confusion, axis=1) + np.sum(confusion, axis=0) -
                    np.diag(confusion))
    _,_,F = F_Score(confusion)

    
    total_res = total_res / dataset_size
    total_res[0] = total_res[0]*100
    total_res[3] = total_res[3]*100
    print(f'all_iou_fscore for dara {data_used} is {mIoU[1]:.4f}_{F[1]:.3f} from {model_name}_upper_{upper_per}')
    print(f'generation_pipeline mse:{total_res[0]:.6f} psnr:{total_res[1]:.6f} AGE: {total_res[2]:.6f} mssim:{total_res[3]:.6f} pEPS:{total_res[4]:.6f} pCEPs:{total_res[5]:.6f}' )
    print(f'{total_res[0]:.2f}_{total_res[1]:.2f}_{total_res[2]:.2f}_{total_res[3]:.2f}_{total_res[4]:.4f}_{total_res[5]:.4f}')



    
        





