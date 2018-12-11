# 3DR
Collection of tools ranging from image alignment, panorama generation, point cloud generation, optical flow estimation to single view 3D reconstruction

<p align="center"><img width="100%" src="imgs/point_cloud.png"/></p>

---
## TODO
- [x] Image Alignment
- [x] Panorama stitching
- [x] Stereo Stitching (Estimate Fundamental/Essential matrix)
- [x] Estimate scene structure (point cloud generation)
- [ ] Custom implementations of (F estimation, triangulation)
- [x] Bundle Adjustment
- [x] SLAM graph optimization

### Bugs (to be fixed)
- ~~point cloud is not dense enough~~ (turned out opencv's triangulation isn't that accurate) 

## Requirements
- C++14
- Linux (tested only on 18.04 LTS)
- OpenCV
- glad
- glfw
- OpenGL
- pangolin
