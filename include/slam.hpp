#ifndef _SLAM_HPP_
#define _SLAM_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <iomanip>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include "viewer.hpp"
#include "frame.hpp"
#include "two.hpp"
#include "map.hpp"
#include "point.hpp"
#include "optimizer.hpp"
#include "camera.hpp"
#include "timer.hpp"

#include <thread>

namespace dr3 {

class SLAM {
private:
   // Image dimensions
   size_t H, W;

   // camera intrinsics
   cv::Mat K;
   cv::Mat I3x4;

   // viewers
   Viewer2D *v2d;
   Viewer3D *v3d;

   // ID of the current frame processed
   static long unsigned int cidx;

   // Abstract camera
   AbstractCamera *cam;

   // previous frame
   Frame *prev_f;

   // render the point cloud in a separate thread
   std::thread render_loop;

   // Map consisting of all the points
   Map *mapp;

   // Tracking time for different operations
   Monitor *monitor;

protected:
   void pprint(Frame &frame);

public:
   SLAM();
   SLAM(size_t H, size_t W,
       cv::Mat K,
       int argc, char **argv);
   ~SLAM();

   void process(cv::Mat &img);
};

} // namespace dr3

#endif
