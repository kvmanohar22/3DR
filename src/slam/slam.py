#!/home/kv/anaconda3/envs/cv/bin/python

import cv2
import numpy as np
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac

H = 1080//2
W = 1920//2

N = np.array([[2./W, 0, -1], [0, 2./H, -1], [0, 0, 1]])
Ninv = np.linalg.inv(N)
orb = cv2.ORB_create()

frames = []

# convert [[x, y]] -> [[x, y, 1]]
def add_ones(x):
  return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

# Draw keypoints
def drawkps(img, kps, color=(0, 0, 255)):
  for kp in kps:
    cv2.circle(img, (int(kp[0]), int(kp[1])), radius=2, color=color)

def drawlines(img, f1, f2, idx1, idx2, color=(0, 0, 255)):
  for i1, i2 in zip(idx1, idx2):
    p1 = tuple(f1.kpus[i1].astype(np.int32))
    p2 = tuple(f2.kpus[i2].astype(np.int32))
    cv2.line(img, p1, p2, color=color, thickness=2)

def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.match(f1.des, f2.des)
  matches = np.array(sorted(matches, key=lambda x: x.distance))
 
  # extract the points which match
  idx1 = np.array([k.queryIdx for k in matches])
  idx2 = np.array([k.trainIdx for k in matches])

  # Use the fundamental matrix to remove outliers
  pts1, pts2 = f1.kpns[idx1], f2.kpns[idx2]
  model, inliers = ransac((pts1, pts2),
                    FundamentalMatrixTransform,
                    min_samples=8, residual_threshold=.005,
                    max_trials=300)

  idx1, idx2 = idx1[inliers], idx2[inliers]
  return idx1, idx2

def extract(img):
  # extract good keypoints
  kpts = cv2.goodFeaturesToTrack(np.mean(img, axis=-1).astype(np.uint8),
              3000, qualityLevel=0.01, minDistance=3)
  kpts = np.squeeze(kpts)
  kptsK = [cv2.KeyPoint(x=kp[0], y=kp[1], _size=20) for kp in kpts]
  kps, des = orb.compute(img, kptsK)
  return np.array([[kp.pt[0], kp.pt[1]] for kp in kps]), des

class Frame(object):
  '''
  kpus: Unnormalized keypoints
  kpns: Normalized keypoints
  '''
  def __init__(self, frame):
    self.h, self.w = frame.shape[:2]
    self.kpus, self.des  = extract(frame) 
    self.kpns = np.dot(N, add_ones(self.kpus).transpose()).transpose()[:, :-1]

def process_frame(frame):
  frame = cv2.resize(frame, (W, H))
  fr = Frame(frame)
  frames.append(fr)  
 
  if len(frames) < 2:
    return

  f1 = frames[-1]
  f2 = frames[-2]
  idx1, idx2 = match_frames(f1, f2)
  
  drawkps(frame, f1.kpus, color=(0, 0, 255))
  drawkps(frame, f2.kpus, color=(0, 255, 0))
  drawlines(frame, f1, f2, idx1, idx2, color=(255, 0, 0))

  cv2.imshow('SLAM', frame)
  cv2.waitKey(20)

if __name__ == '__main__':
  cap = cv2.VideoCapture("../../videos/test.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
