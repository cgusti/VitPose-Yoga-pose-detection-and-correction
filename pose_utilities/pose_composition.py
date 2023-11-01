import numpy as np 

class BodyPart: 
    def __init__(self, body_part):
        """
        Constructor for 1 body part consisting of x, y coordinates, confidence, and boolean exists information 
        Parameters: 
        body_part: int 
                   1 x 3 ndarray consisteing of x, y, confidence level
        """
        self.x = body_part[0]
        self.y = body_part[1]
        self.c = body_part[2]
        self.exists = self.c != 0 #check if body part exists or not within one frame
     
    def __truediv__(self, scalar):
        return BodyPart([self.x/scalar, self.y/scalar, self.c])
    
    def __floordiv__(self, scalar):
        __truediv__(self, scalar)
            
    @staticmethod
    def dist(bodypart1, bodypart2):
        "Calculates the Euclidean distance between BodyPart instances"
        return np.sqrt(np.square(bodypart1.x - bodypart2.x) + np.square(bodypart1.y - bodypart2.y))
    

class Pose:
    #based on COCO dataset
    BODY_PART_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 
                       'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                       'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
                       'left_ankle', 'right_ankle'] 
    def __init__(self, body_parts):
        """
        Constructs a pose for one frame, given an array of parts 
        Parameters: 
            body_parts: 17 x 3 ndarray of x, y, confidence values
        """
        for name, vals in zip(self.BODY_PART_NAMES, body_parts):
            setattr(self, name, BodyPart(vals))

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        out = ''
        for name in self.BODY_PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ + '\n'
            return out
    
    def print(self, parts):
        """
        Print x and y coordinates of body parts on the current instance of the class
        Parameters: 
            parts: list 
                   list containing sequence of body parts E.g. ['nose', 'left ear']
        Returns: 
            x and y coordinates of each bodypart as set in construction time
        """
        out = ''
        # out = dict()
        for name in parts:
            if not name in self.BODY_PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).y)
            out = out + _ + ','
            # out[name] = (getattr(self, name).x, getattr(self, name).y)
        return out
    
class PoseSequence:
    """Contains a sequence of chained Pose Objects + normalization 
    based on average torso length"""
    def __init__(self, sequence):
        """
        Parameters: 
        sequence: list of nx17x3 arrays, where n equal number of frames for a pose 
        """
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # print(f'self.poses after uploading frames: {self.poses}')
        #normalize poses based on the average pixel length
        torso_lengths = np.array([BodyPart.dist(pose.nose, pose.left_hip) for pose in self.poses if pose.nose.exists and pose.left_hip.exists] +
                                 [BodyPart.dist(pose.nose, pose.right_hip) for pose in self.poses if pose.nose.exists and pose.right_hip.exists])     
        mean_torso = np.mean(torso_lengths)   
        
        for pose in self.poses: 
            for attr, part in pose: 
                setattr(pose, attr, part/mean_torso)  
                # print('progression of left and right hips:', {pose.print(['left_hip', 'right_hip'])})


