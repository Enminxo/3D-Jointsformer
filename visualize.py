
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle


def plot_frame(vid, label, edges=[], idxs=[]):
    # get x,y coordinates
    x = list(vid[..., 0])
    y = list(vid[..., 1])

    ax.clear()
    ax.scatter(x, y, color='dodgerblue')
    for i in range(vid.shape[0]):
        ax.text(x[i], y[i], idxs[i])

    for edge in edges:
        ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color='salmon')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(label)


# todo: 1. First set up the figure, the axis, and the plot element we want to animate
#       2. animation function.  This is called sequentially
#       3. call the animator.

# edges defined according to mediapipe hands model : https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (0, 17), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20)]

# load the pickle file
with open('data/briareo/RGB/test.pkl', 'rb') as fp:
    test_data = pickle.load(fp)  # list of dict

fig, ax = plt.subplots()
vid_id = 20
vid = test_data[vid_id]['keypoint']  # ndarray (57,21,3)
vid_label = test_data[vid_id]['video_id']  # ndarray (57,21,3)
vid_idxs = list(range(vid.shape[0]))  # len_vid
# Animate the frames

anim = FuncAnimation(fig,
                     lambda frame: plot_frame(frame,vid_label, edges, vid_idxs),
                     frames=vid,
                     interval=200)
plt.show()
