import pangolin as pgl
import OpenGL.GL as gl
import numpy as np
import cv2

from utils import drawkps
from utils import drawlines
from utils import extract
from utils import add_ones
from utils import N

class Display3D(object):
  def __init__(self, H, W):
    self.cameras = []
    self.points = []
    self.viewer_init(H, W)

  def viewer_init(self, H, W):
    pgl.CreateWindowAndBind('SLAM', W, H)
    gl.glEnable(gl.GL_DEPTH_TEST) 
    self.scam = pgl.OpenGlRenderState(
            pgl.ProjectionMatrix(W, H, 420, 420, W//2, H//2, 0.2, 100), 
            pgl.ModelViewLookAt(0, -10, -8, 
                                0, 0, 0, 
                                0, -1, 0
                                ))
    self.handler = pgl.Handler3D(self.scam)

    self.dcam = pgl.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, (-1.0 * W)/H)
    self.dcam.SetHandler(self.handler)

  def add_observation(self, camera, points):
    self.cameras.append(camera)
    for point in points:
      self.points.append(point)

  def refresh(self):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)

    # Relative poses
    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pgl.DrawCameras(self.cameras)

    # Point Cloud
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pgl.DrawPoints(self.points)

    pgl.FinishFrame()

class Display2D(object):
  def __init__(self):
    pass

  def refresh(self, frame, f1, f2, idx1, idx2):
    # Draw keypoints
    drawkps(frame, f1.kpus, color=(0, 255, 0))
    drawkps(frame, f2.kpus, color=(255, 0, 0))
    
    # Draw matching lines
    drawlines(frame, f1, f2, idx1, idx2, color=(0, 0, 255))
    
    # Update view
    cv2.imshow('SLAM', frame)
    cv2.waitKey(20)

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

