
## Inference

import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed
from models.model import GPTConfig,GPT
import argparse
from utils.util import sample_mask,sample_mask_all
from tqdm import tqdm
from PIL import Image
import os
import io 
from cv2 import cv2 
from zipfile import ZipFile 
import time

if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument('--split',type=int, default=0)
    parser.add_argument('--level',type=int, default=0)
    parser.add_argument('--total',type=int, default=1)
    parser.add_argument('--GPU_ids',type=str,default='0')
    parser.add_argument('--ckpt_path',type=str,default='./ckpt')
    parser.add_argument('--BERT',action='store_true', help='BERT model, Image Completion')
    parser.add_argument('--image_url',type=str,default='',help='the folder of image')
    parser.add_argument('--mask_url',type=str,default='',help='the folder of mask')
    parser.add_argument('--top_k',type=int,default=100)

    parser.add_argument('--image_size',type=int,default=32,help='input sequence length: image_size*image_size')

    parser.add_argument('--n_layer',type=int,default=14)
    parser.add_argument('--n_head',type=int,default=8)
    parser.add_argument('--n_embd',type=int,default=256)
    parser.add_argument('--GELU_2',action='store_true',help='use the new activation function')

    parser.add_argument('--save_url',type=str,default='./',help='save the output results')
    parser.add_argument('--n_samples',type=int,default=8,help='sample cnt')

    parser.add_argument('--sample_all',action='store_true',help='sample all pixel together, ablation use')
    parser.add_argument('--skip_number',type=int,default=0,help='since the inference is slow, skip the image which has been inferenced')

    parser.add_argument('--no_progressive_bar',action='store_true',help='')
    # parser.add_argument('--data_path',type=str,default='/home/ziyuwan/workspace/data/')

    opts=parser.parse_args()

    s_time=time.time()

    # model_config=GPTConfig(512,32*32,
    #                        embd_pdrop=0.0, resid_pdrop=0.0, 
    #                        attn_pdrop=0.0, n_layer=14, n_head=8,
    #                        n_embd=256,BERT=opts.BERT)

    model_config=GPTConfig(512,opts.image_size*opts.image_size,
                           embd_pdrop=0.0, resid_pdrop=0.0, 
                           attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head,
                           n_embd=opts.n_embd, BERT=opts.BERT, use_gelu2=opts.GELU_2)

    # Load model
    IGPT_model=GPT(model_config)
    checkpoint=torch.load(opts.ckpt_path)
    
    if opts.ckpt_path.endswith('.pt'):
        IGPT_model.load_state_dict(checkpoint)
    else:
        IGPT_model.load_state_dict(checkpoint['model'])

    IGPT_model.cuda()

    # Load clusters
    C = np.load('kmeans_centers.npy') ## [0,1]
    C = np.rint(127.5 * (C + 1.0))
    C = torch.from_numpy(C)

    n_samples=opts.n_samples

    img_list=sorted(os.listdir(opts.image_url))
    # mask_list=sorted(os.listdir(opts.mask_url))
    # mask reading from pconv.zip 
    mask_list = [f'pconv/{str(2000 * opts.level + i).zfill(5)}.png' for i in range(2000)]
    mask_list = mask_list * int(len(img_list) / len(mask_list) + 1)
    # set subset 
    img_list = [img_list[i] for i in range(opts.split, len(img_list), opts.total)]
    mask_list = [mask_list[i] for i in range(opts.split, len(mask_list), opts.total)]
    
    if opts.skip_number>0:
        img_list=img_list[opts.skip_number-1:]
        mask_list=mask_list[opts.skip_number-1:]
        print("Resume from %s"%(img_list[0]))


    if opts.BERT:
        os.makedirs(os.path.join(opts.save_url,'masked'), exist_ok=True)
        for x_name,y_name in tqdm(zip(img_list,mask_list), total=len(img_list)):

            # if x_name!=y_name:
            #     print("### Something Wrong ###")

            # mask_url=os.path.join(opts.mask_url,y_name)
            # input_mask=Image.open(mask_url).convert("L")
            input_mask = ZipFile(opts.mask_url).read(y_name)
            input_mask = Image.open(io.BytesIO(input_mask)).convert('L')
            image_url=os.path.join(opts.image_url,x_name)
            input_image = Image.open(image_url).convert("RGB")
            input_image = np.array(input_image) * (1.0 - np.expand_dims(np.array(input_mask), axis=2)/255.0)
            input_image = Image.fromarray(input_image.astype(np.uint8))
            input_image.save(os.path.join(opts.save_url,'masked', x_name))

            x = input_image.resize((opts.image_size,opts.image_size),resample=Image.BILINEAR)
            x = torch.from_numpy(np.array(x)).view(-1, 3)
            x = x.float()
            a = ((x[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1) # cluster assignments

            # dilate mask 
            input_mask = cv2.dilate(np.array(input_mask), np.ones((21,21), 'uint8'), iterations=1)
            input_mask[input_mask > 0] = 255 
            input_mask = Image.fromarray(input_mask.astype(np.uint8)).convert('L')
            y = input_mask.resize((opts.image_size,opts.image_size), resample=Image.NEAREST)
            y = torch.from_numpy(np.array(y)/255.).view(-1)
            # y = y>0.5
            y = y > 0.1
            y = y.float()

            a_list=[a]*n_samples
            a_tensor=torch.stack(a_list,dim=0) ## Input images
            b_list=[y]*n_samples
            b_tensor=torch.stack(b_list,dim=0) ## Input masks
            a_tensor*=(1-b_tensor).long()

            if opts.sample_all:
                pixels=sample_mask_all(IGPT_model,context=a_tensor,length=opts.image_size*opts.image_size,num_sample=n_samples,top_k=opts.top_k,mask=b_tensor,no_bar=opts.no_progressive_bar)
            else:
                pixels=sample_mask(IGPT_model,context=a_tensor,length=opts.image_size*opts.image_size,num_sample=n_samples,top_k=opts.top_k,mask=b_tensor,no_bar=opts.no_progressive_bar)

            img_name=x_name[:-4]+'.png'
            for i in range(n_samples):
                current_url=os.path.join(opts.save_url,'condition_%d'%(i+1))
                os.makedirs(current_url,exist_ok=True)
                current_img=C[pixels[i]].view(opts.image_size, opts.image_size, 3).numpy().astype(np.uint8)
                tmp=Image.fromarray(current_img)
                tmp.save(os.path.join(current_url,img_name))
        
        e_time=time.time()
        print("This test totally costs %.5f seconds"%(e_time-s_time))
