import numpy as np
import g2o

from utils import get_pose

class Optimizer(object):
  def __init__(self, mapp):
    self.mapp = mapp

  def optimize(self):
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)
       
    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

    # add frames to graph
    for f in self.mapp.cameras:
      sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[:3, :3], f.pose[:3, 3]))
      # sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[2][0], f.K[2][1], 1.0)
      sbacam.set_cam(1.0, 1.0, 0.0, 0.0, 1.0)

      v_se3 = g2o.VertexCam()
      v_se3.set_id(f.idx)
      v_se3.set_estimate(sbacam)
      v_se3.set_fixed(False)
      opt.add_vertex(v_se3)

    # add points to graph
    PI_ID_OFFSET = 0x10000
    for p in self.mapp.points:
      pt = g2o.VertexSBAPointXYZ()
      pt.set_id(p.idx + PI_ID_OFFSET)
      pt.set_estimate(p.xyz[:3])
      pt.set_marginalized(True)
      pt.set_fixed(False)
      opt.add_vertex(pt)

      for f in p.frames:
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, pt)
        edge.set_vertex(1, opt.vertex(f.idx))
        uv = f.kpns[f.pts.index(p)]
        edge.set_measurement(uv)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(robust_kernel)
        opt.add_edge(edge)

    opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(10)
       
    # put frames back
    for f in self.mapp.cameras:
      est = opt.vertex(f.idx).estimate()
      R = est.rotation().matrix()
      t = est.translation()
      f.pose = get_pose(R, t)

    # put points back
    for p in self.mapp.points:
      est = opt.vertex(PI_ID_OFFSET + p.idx).estimate()
      p.xyz = np.array(est)

