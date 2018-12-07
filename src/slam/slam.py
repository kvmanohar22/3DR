#!/home/kv/anaconda3/envs/cv/bin/python

import cv2

H = 1080//2
W = 1920//2

def process_frame(frame):
  frame = cv2.resize(frame, (W, H))
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
