import matplotlib as plt


def plot_keypoints(coordinates):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    
    # labels for each point
    keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", 
             "right_shoulder", "left_elbow","right_elbow", "left_wrist", "right_wrist","left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    # plot 
    plt.scatter(x, y)

    # Add labels to each point
    for coord, label in zip(coordinates, keypoints):
        plt.text(coord[0], coord[1], label, ha='center', va='bottom')
        
    # Set the plot title and labels
    plt.title('Scatter Plot with Point Labels')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Display the plot
    plt.show()