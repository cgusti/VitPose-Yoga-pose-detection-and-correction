"""Takes in a video split it into different image frames"""

"""This is unfinished"""
import argparse
import cv2
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

# def main(): 
#     parser = argparse.ArgumentParsere(description='Video Pre-Processing')
#     parser.add_argument('--input_video', type=str, default='', help='input video to be split into individual frames')
#     parser.add_argument('--output_folder', type=str, default='', help='output directory to contain image frames')
#     parser.add_argument('--yoga_pose_name', type=str)
#     parser.add_argument('--correct', type=str, default='', help='Whether video is a correct or wrong yoga pose')
#     args= parser.parse_args()

#     load_video(args.input_video, args.output_folder, yoga_pose_name, correct)
    

def load_video(video_path, output_folder, fps=3):
    """loads a video from and splits into different image frames based on frames per second"""
    capture = cv2.VideoCapture(video_path)
    
    #get original video's frame rate
    original_fps = capture.get(cv2.CAP_PROP_FPS)
    
    #calculate the frame interval based on the desired fps
    frame_interval = int(round(original_fps / fps))

    #create output dir if it doesn't exist 
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    frame_index = 0
    
    while True:
        success, frame = capture.read()
        
        if not success:
            break
        
        #Process the frame only if it's within the desired frame interval 
        if frame_index % frame_interval == 0:
            output_file = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_file, frame)
            frame_count += 1
        frame_index += 1
    capture.release()

def split_video_frames(input_file, output_dir, yoga_pose_name, frame_rate):
    vidcap = cv2.VideoCapture(input_file)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(f"{output_dir}/{yoga_pose_name}_{count}.jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        success = getFrame(sec)

# Example usage: 
# video_path = '/home/cgusti/ViTPose_pytorch/data/input_poses/Downward Dog.mp4'
# output_folder = '/home/cgusti/ViTPose_pytorch/data/output'
# yoga_pose_name = 'downward_dog'
# split_video_frames(video_path, output_folder, 'downward_dog', 1)

    