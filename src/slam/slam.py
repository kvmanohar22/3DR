#!/home/kv/anaconda3/envs/cv/bin/python

import cv2
import numpy as np

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

def drawlines(img, pts1, pts2, color=(0, 0, 255)):
  for p1, p2 in zip(pts1, pts2):
    cv2.line(img, p1, p2, color=color)

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
    self.kpns = np.dot(N, add_ones(self.kpus).transpose()).transpose()

def process_frame(frame):
  frame = cv2.resize(frame, (W, H))
  fr = Frame(frame)
  frames.append(fr)  
 
  if len(frames) < 2:
    return

  f1 = frames[-1]
  f2 = frames[-2]
  
  drawkps(frame, f1.kpus, color=(0, 0, 255))
  drawkps(frame, f2.kpus, color=(0, 255, 0))

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
