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

def get_pose(R, t):
  Rt = np.eye(4)
  Rt[:3, :3] = R
  Rt[:3, -1] = t
  return Rt

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
    cv2.line(img, p1, p2, color=color, thickness=1)

def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)
 
  idx1, idx2 = [], []
  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      idx1.append(m.queryIdx)
      idx2.append(m.trainIdx)

  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  # Use the fundamental matrix to remove outliers
  pts1, pts2 = f1.kpns[idx1], f2.kpns[idx2]
  model, inliers = ransac((pts1, pts2),
                    FundamentalMatrixTransform,
                    min_samples=8, residual_threshold=.005,
                    max_trials=300)

  idx1, idx2 = idx1[inliers], idx2[inliers]
  F = np.dot(np.dot(N.transpose(), model.params), N)

  # Extract rotation and translation matrices from F
  W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float)

  U, D, Vt = np.linalg.svd(F)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) < 0:
    Vt *= -1;
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, -1]
  return idx1, idx2, get_pose(R, t)

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
  pose: (R, t) matrix relative to the first frame
  '''
  def __init__(self, frame):
    self.h, self.w = frame.shape[:2]
    self.kpus, self.des  = extract(frame) 
    self.kpns = np.dot(N, add_ones(self.kpus).transpose()).transpose()[:, :-1]
    self.pose = np.eye(4)

def process_frame(frame):
  frame = cv2.resize(frame, (W, H))
  fr = Frame(frame)
  frames.append(fr)  
 
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

  # Draw the keypoints and matches
  drawkps(frame, f1.kpus, color=(0, 255, 0))
  drawkps(frame, f2.kpus, color=(255, 0, 0))
  drawlines(frame, f1, f2, idx1, idx2, color=(0, 0, 255))
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
