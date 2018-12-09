import pangolin as pgl
import OpenGL.GL as gl
import numpy as np
import cv2

from utils import drawkps
from utils import drawlines
from utils import extract
from utils import add_ones
from utils import K, Kinv

class Display3D(object):
  def __init__(self):
    self.cameras = []
    self.points = []
    self.H, self.W = 900, 1600
    self.viewer_init()

  def viewer_init(self):
    pgl.CreateWindowAndBind('SLAM', self.W, self.H)
    gl.glEnable(gl.GL_DEPTH_TEST) 
    self.scam = pgl.OpenGlRenderState(
            pgl.ProjectionMatrix(self.W, self.H, 420, 420, self.W//2, self.H//2, 0.2, 100), 
            pgl.ModelViewLookAt(0, -10, -8, 
                                0, 0, 0, 
                                0, -1, 0
                                ))
    self.handler = pgl.Handler3D(self.scam)

    self.dcam = pgl.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, self.W/self.H)
    self.dcam.SetHandler(self.handler)

  def refresh(self):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)
    
    # get the latest data
    cameras = []
    for cam in self.cameras:
      cameras.append(cam.pose)
    xyzs, cols = [], []
    for xyz in self.points:
      xyzs.append(xyz.xyz)
      cols.append(xyz.col)

    xyzs = np.array(xyzs)
    cols = np.array(cols)
    
    # Relative poses
    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pgl.DrawCameras(cameras)

    # Point Cloud
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pgl.DrawPoints(xyzs, cols/256.0)

    pgl.FinishFrame()

class Display2D(object):
  def __init__(self):
    pass

  def refresh(self, frame, f1, f2, idx1, idx2):
    # Draw keypoints
    drawkps(frame, f1.kpus[idx1], color=(0, 255, 0))
    drawkps(frame, f2.kpus[idx2], color=(255, 0, 0))
    
    # Draw matching lines
    drawlines(frame, f1, f2, idx1, idx2, color=(0, 0, 255))
    
    # Update view
    cv2.imshow('SLAM', frame)
    cv2.waitKey(20)

class Frame(object):
  '''
  kpus: Unnormalized keypoints
  kpns: Normalized keypoints
  pose: (R, t) matrix transformation relative to first frame
  '''
  def __init__(self, display3d, frame):
    self.h, self.w = frame.shape[:2]
    self.kpus, self.des  = extract(frame) 
    self.kpns = np.dot(Kinv, add_ones(self.kpus).T).T[:, :2]
    self.pose = np.eye(4)
    display3d.cameras.append(self)

class Point(object):
  def __init__(self, display3d, pt, col):
    self.xyz = np.copy(pt)
    self.col = np.copy(col)
    display3d.points.append(self)

