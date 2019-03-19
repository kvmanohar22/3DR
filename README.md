# 3DR
Collection of tools ranging from image alignment, panorama generation, point cloud generation, optical flow estimation to single view 3D reconstruction.

<p align="center"><img width="100%" height="50%" src="imgs/slam/point_cloud.png"/></p>
Tracking camera (without Bundle Adjustment)

---
## TODO
- [x] Image Alignment
- [x] Image Warping
- [x] Panorama stitching
- [ ] Ghost removal in big panoramas (refer [1])
- [ ] Optical Flow
- [ ] SfM (unordered images)
- [ ] SfM (video sequence (localisation))

## Requirements
- C++14
- Linux (tested only on Ubuntu 18.10)
- OpenCV (for image I/O)
- ceres-solver (for Bundle Adjustment)
- Pangolin (for 3D viewer)

## References
1. M. Uyttendaele, A. Eden, and R. Szeliski.
    Eliminating ghosting and exposure artifacts in image mosaics.
    In Proceedings of the Interational Conference on Computer Vision and Pattern Recognition
