import pangolin as pgl
import OpenGL.GL as gl
import numpy as np
import cv2

from utils import drawkps
from utils import drawlines
from utils import extract
from utils import add_ones
from utils import N, Ninv

from multiprocessing import Queue, Process

class Display3D(object):
  def __init__(self):
    self.cameras = []
    self.points = []
    self.q = None

  def create_viewer(self):
    self.q = Queue()
    self.viewer = Process(target=self.viewer_thread, args=(self.q, ))
    self.viewer.start()

  def viewer_thread(self, q):
    self.viewer_init(1600, 900)
    while True:
      self.refresh(q) 

  def viewer_init(self, w, h):
    pgl.CreateWindowAndBind('SLAM', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST) 
    self.scam = pgl.OpenGlRenderState(
            pgl.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 100), 
            pgl.ModelViewLookAt(0, -10, -8, 
                                0, 0, 0, 
                                0, -1, 0
                                ))
    self.handler = pgl.Handler3D(self.scam)

    self.dcam = pgl.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
    self.dcam.SetHandler(self.handler)

  def refresh(self, q):
    if not q.empty():
      return
       
    cameras, xyzs, cols = q.get()  

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)
    
    # Relative poses
    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pgl.DrawCameras(cameras)

    # Point Cloud
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pgl.DrawPoints(xyzs, cols)

    pgl.FinishFrame()

  def updateQ(self):
    if self.q is None:
      return

    cameras, xyzs, cols = [], [], []
    for cam in self.cameras:
      cameras.append(cam.pose)
    for xyz in self.points:
      xyzs.append(xyz.xyz)
      cols.append(xyz.col)

    cameras = np.array(cameras)
    xyzs = np.array(xyzs)
    cols = np.array(cols)/256.0

    self.q.put((cameras, xyzs, cols))

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
    self.kpns = np.dot(N, add_ones(self.kpus).T).T[:, :2]
    self.pose = np.eye(4)
    display3d.cameras.append(self)
    self.idx = len(display3d.cameras)
    self.pts = [None] * len(self.kpns)

class Point(object):
  '''
  xyz   : 3D location of the point
  col   : col of the point
  idx   : unique id of the point
  
  frames: Different frames from which this point was observed in
  idxs  : unique ids of the frames
  '''
  def __init__(self, display3d, pt, col):
    self.xyz = np.copy(pt)
    self.col = np.copy(col)
    display3d.points.append(self)
    self.idx = len(display3d.points)
    self.frames = []
    self.idxs = []

  def add_observation(self, frame, idx):
    '''
    This particular point was seen in this frame with given idx
    '''
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)

