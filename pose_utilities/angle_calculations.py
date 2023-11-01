import numpy as np
import os
import math

def calculate_angles(vector_series_1, vector_series_2):
    """Calculate the angle using dot product for each pair of rows in two arrays. """
    dot_products = np.sum(vector_series_1 * vector_series_2, axis=1)
    magnitudes_1 = np.linalg.norm(vector_series_1, axis=1)
    magnitudes_2 = np.linalg.norm(vector_series_2, axis=1)    
    #calculate angles in radians
    angles_rad = np.arccos(dot_products/(magnitudes_1 * magnitudes_2))
    #convert angles to degress
    angles_deg = np.degrees(angles_rad)
    return angles_deg


def calculate_angle_with_horizontal(vector_series):
    x_coords = vector_series[:,0]
    y_coords = vector_series[:,1]
    angles = []
    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]
        angle = math.degrees(np.arctan2(y, x))
        angles.append(angle)
    return angles


def load_features_warriorTwo(file_paths, output_dir):
    """
    load necessary features for a Warrior Two yoga pose
    Params: 
        inout file (list): list of input file paths
    Returns: 
        list of numpy array sequences for engineered features 
    """ 
    bent_leg_angles = []
    torso_bend_angles = []
    hip_bend_angles = []
    
    for filepath in file_paths: 
        ps = load_ps(filepath) #loading pose sequence object 
        poses_sequence = ps.poses #load all poses in the sequence 
        
        pose_sequence_body_parts = [(pose.left_shoulder, pose.right_shoulder, pose.left_elbow, pose.right_elbow, pose.left_wrist, pose.right_wrist, 
                   pose.left_hip, pose.right_hip, pose.left_knee, pose.right_knee, pose.left_ankle, pose.right_ankle) for pose in poses_sequence]
        
        #filter out data points/frames where a part does not exist. We only want full, usable frames 
        filtered_sequence = [body_pose for body_pose in pose_sequence_body_parts if all(part.exists for part in body_pose)]
        
        #calculate left bend angle 
        left_upper_leg_vectors = np.array([(body_parts[6].x - body_parts[8].x, body_parts[6].y - body_parts[8].y) for body_parts in filtered_sequence])
        left_lower_leg_vectors = np.array([(body_parts[8].x - body_parts[10].x, body_parts[8].y - body_parts[10].y) for body_parts in filtered_sequence])
        left_leg_bend = calculate_angles(left_upper_leg_vectors, left_lower_leg_vectors)
        left_leg_bend_filtered = medfilt(medfilt(left_leg_bend, 5),5)
        # print(f"left leg bend: {left_leg_bend}")
        
        #calculate right bend angle 
        right_upper_leg_vectors = np.array([(body_parts[7].x - body_parts[9].x, body_parts[7].y - body_parts[9].y) for body_parts in filtered_sequence])
        right_lower_leg_vectors = np.array([(body_parts[9].x - body_parts[11].x, body_parts[9].y - body_parts[11].y) for body_parts in filtered_sequence])
        right_leg_bend = calculate_angles(right_upper_leg_vectors, right_lower_leg_vectors)
        #use a median filter to eliminate outliers 
        right_leg_bend_filtered = medfilt(medfilt(right_leg_bend, 5),5)
        
        #deduce which side person is facing
        side = 'right' if np.mean(right_leg_bend_filtered) > np.mean(left_leg_bend_filtered) else 'left'
        
        if side == 'right': 
            print('right is bent leg')
            bent_leg_angle = right_leg_bend_filtered
            bent_leg_angles.append(bent_leg_angle)
        else: 
            print('left is bent leg')
            bent_leg_angle = left_leg_bend_filtered
            bent_leg_angles.append(bent_leg_angle)
        
        #calculate arc tangent for hips 
        hip_vectors = np.array([(body_parts[6].x - body_parts[7].x, body_parts[6].y - body_parts[7].y) for body_parts in filtered_sequence])
        hip_bend_angle = calculate_angle_with_horizontal(hip_vectors)
        hip_bend_angles.append(hip_bend_angle)
        
        #calculate left/right torso angle with the horizontal 
        torso_vectors = np.array([(body_parts[0].x - body_parts[11].x, body_parts[0].y - body_parts[11].y) for body_parts in filtered_sequence])
        torso_bend_angle = calculate_angle_with_horizontal(torso_vectors)
        torso_bend_angles.append(torso_bend_angle)

        #save recorded angles in numpy filre 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = os.path.splitext(filepath)[0].split('/')[-1]
        np.savez(os.path.join(output_dir, file_name), bent_leg=bent_leg_angle, torso_bend=torso_bend_angle, hip_bend=hip_bend_angle)
    
    return bent_leg_angles, torso_bend_angles, hip_bend_angles

#Note: decided not to include straight_leg_angles because of inconsistencies