from scripts.Utilities.pose_composition import PoseSequence
import numpy as np

def load_ps(filename):
    """Load a PoseSequence object from a given numpy file. 
    
    Args: 
        filename: file name of the numpy file containing keypoints
        
    Returns: 
        PoseSequence object with normalized joint keypoints
    """
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)