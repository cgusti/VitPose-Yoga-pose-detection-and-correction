import argparse
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np
from time import time
from PIL import Image
from torchvision.transforms import transforms

import sys
import os
from models.model import ViTPose
from model_utilities.visualization import draw_points_and_skeleton, joints_dict
from model_utilities.dist_util import get_dist_info, init_dist
from model_utilities.top_down_eval import keypoints_from_heatmaps

__all__ = ['inference']       
            
@torch.no_grad() #disabling gradient calculation (typically used in situations where the model parameters are fixed and there is no need to compute gradients, such as during inference)
def inference(img_path: Path, output_path: Path, img_size: Tuple[int, int],
              model_cfg: dict, ckpt_path: Path, device: torch.device, save_result: bool=True) -> np.ndarray:
    '''
    Outputs: 
    (1) points: is a 3 dimensional array with shape (1, num_keypoints, 2), where num_keypoints = # number of keypoints the model is trained to detect
    Each entry represents the (x,y) coordinates of the first detected keypoint (E.g [343.3852])
    (2) prob: is a 1 x num_keypoints matrix where each entry represents the probability of a particular keypoint being present at a given pixel location in the input image 
    '''
    
    # Prepare model
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)
    print(f">>> Model loaded: {ckpt_path}")
    
    # Prepare input data
    img = Image.open(img_path)
    org_w, org_h = img.size
    # print(f">>> Original image size: {org_h} X {org_w} (height X width)")
    # print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
    # print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")
    img_tensor = transforms.Compose (
        [transforms.Resize((img_size[1], img_size[0])),
         transforms.ToTensor()]
    )(img).unsqueeze(0).to(device)
    
    # Feed to model
    tic = time()
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
    elapsed_time = time()-tic
    # print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
    
    
    # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                           unbiased=True, use_udp=True)
    points = np.concatenate([points[:, :, ::-1], prob], axis=2) #concatenate points with probabilities (1, num_keypoints, heatmap_size)
    # print(f'printing final points: {points}')
    
    #Visualizaiton
    if save_result:
        for pid, point in enumerate(points):
            img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)
    return points, img
    