from configs.ViTPose_base_coco_256x192 import model as model_cfg
from configs.ViTPose_base_coco_256x192 import data_cfg
import os
import torch
import re
import glob
import numpy as np

def parse_sequence(input_folder, output_folder_keypoints, output_folder_visual):
    """
    Parse a sequence of image frames and saves each corresponding 
    pose from each image frame as a numpy file within the output directory
    
    Parameters: 
        input_folder: str 
                      path to the folder containing spliced image frames for one video
        output_folder: str 
                      path to saved numpy array files of keypoints 
    """
    
    file_names = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"file_names: {file_names}")
    num_frames = len(file_names)
    print(f"num_frames: {num_frames}")
    all_keypoints = np.zeros((num_frames, 17, 3))
    
    #set variables for model inference
    CKPT_PATH = "/home/cgusti/ViTPose_pytorch/scripts/checkpoints/vitpose-b-multi-coco.pth"
    img_size = data_cfg['image_size']
    
    for i, file_name in enumerate(file_names):
        img_path = input_folder + '/' + file_name #e.g data/input_frames/WarriorTwo/WarriorTwo_correct_5/frame_1.jpg
        print(f"image path for inference index {i}: {img_path}")
        #model inference
        keypoints, img_skeleton= inference(img_path=img_path, output_path=output_folder_keypoints, img_size=img_size, 
                              model_cfg=model_cfg, ckpt_path=CKPT_PATH, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                              save_result=True) #this function will automatically save result in output path
        print(f'keypoints that comes out of the model from inference:  {keypoints}')
        all_keypoints[i] = keypoints 
        cv2.imwrite(output_folder_visual + f'_frame{i}.jpg', img_skeleton)
        print(f'current all keypoints: {all_keypoints}')
    np.save(output_folder_keypoints + '_sequence.npy', all_keypoints)