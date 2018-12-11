#!/home/kv/anaconda3/envs/cv/bin/python

import cv2
import os
import numpy as np
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac
import pangolin as pgl
import OpenGL.GL as gl

from core import Frame
from core import Display2D
from core import Display3D
from core import Point
from utils import match_frames, H, W
from optimizer import Optimizer 

frames = []

# core classes
display2d = Display2D() if os.getenv('D2D') is not None else None
display3d = Display3D() 
optimizer = Optimizer(display3d)

# initialize viewer thread
if os.getenv('D3D') is not None:
  display3d.create_viewer()

def triangulate(pose1, pose2, pts1, pts2):
  ret = np.zeros((pts1.shape[0], 4))

  pose1 = np.linalg.inv(pose1)
  pose2 = np.linalg.inv(pose2)

  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4, 4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose1[0]
    A[3] = p[1][1] * pose2[2] - pose1[1]

    _, _, vt = np.linalg.svd(A)
    ret[i] = vt[3]

  return ret

def process_frame(img):
  frame = cv2.resize(img, (W, H))
  fr = Frame(display3d, frame)
  frames.append(fr)  
  print('\n*** frame {} ***'.format(fr.idx))

  if len(frames) < 2:
    return

  f1 = frames[-1]
  f2 = frames[-2]
  idx1, idx2, Rt = match_frames(f1, f2)
  
  # update the pose of the frame
  f1.pose = np.dot(Rt, f2.pose)

  # triangulate the points
  pts4d = triangulate(f1.pose, f2.pose, f1.kpns[idx1], f2.kpns[idx2])
  pts4d /= pts4d[:, 3:]

  for i, idx in enumerate(idx2):
    if f2.pts[idx] is not None:
      f2.pts[idx].add_observation(f1, idx1[i])

  unmatched_pts = np.array([f1.pts[idx] is None for idx in idx1])
  
  # filter points
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_pts 

  # spit out some stats
  print('pts4d: {} -> {:3d}'.format(pts4d.shape[0], pts4d[good_pts4d].shape[0]))
  print('Adding {} points'.format(np.sum(unmatched_pts)))

  for i, xyz in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    u, v = f1.kpus[idx1[i]].astype(np.int32)
    pt = Point(display3d, xyz, frame[v, u])
 
    # Add obervations
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  # optimize the map
  if fr.idx >= 4:
    optimizer.optimize()
    exit(-1)
  
  # 3D display
  display3d.updateQ()
  
  # 2D display
  if display2d is not None:
    display2d.refresh(frame, f1, f2, idx1, idx2)

if __name__ == '__main__':
  cap = cv2.VideoCapture("../../videos/test.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
