#!/home/kv/anaconda3/envs/cv/bin/python

import cv2
import numpy as np
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac
import pangolin as pgl
import OpenGL.GL as gl

from core import Frame
from core import Display2D
from core import Display3D
from utils import match_frames, H, W

frames = []

# core classes
display2d = Display2D()
display3d = Display3D(H, W)

def process_frame(frame):
  frame = cv2.resize(frame, (W, H))
  fr = Frame(frame)
  frames.append(fr)  
  print('\n*** frame {} ***'.format(len(frames)))

  if len(frames) < 2:
    return

  f1 = frames[-1]
  f2 = frames[-2]
  idx1, idx2, Rt = match_frames(f1, f2)
  
  # update the pose of the frame
  f1.pose = np.dot(Rt, f2.pose)

  # triangulate the points
  pts4d = cv2.triangulatePoints(f1.pose[:3], f2.pose[:3], f1.kpus[idx1].T, f2.kpus[idx2].T).T
  pts4d[:, :3] /= pts4d[:, -1:]

  # filter points
  good_pts4d = (np.abs(pts4d[:, 3] > 0.005)) & (pts4d[:, 2] > 0)
  final_pts4d = pts4d[good_pts4d][:, :3]

  print('pts4d: {:3d} -> {:3d}'.format(pts4d.shape[0], final_pts4d.shape[0]))
  # 3D display
  display3d.add_observation(Rt, final_pts4d)
  display3d.refresh()
  
  # 2D display
  display2d.refresh(frame, f1, f2, idx1, idx2)

if __name__ == '__main__':
  cap = cv2.VideoCapture("../../videos/test.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
