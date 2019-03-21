#ifndef _SLAM_HPP_
#define _SLAM_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <iomanip>

#include "viewer.hpp"
#include "frame.hpp"
#include "two.hpp"
#include "map.hpp"
#include "point.hpp"

#include <thread>

namespace dr3 {

class SLAM {
private:
   // Image dimensions
   unsigned int H, W;

   // camera intrinsics
   cv::Mat K;
   cv::Mat I3x4;

   // viewers
   Viewer2D *v2d;
   Viewer3D *v3d;

   // ID of the current frame processed
   static long unsigned int cidx;

   // previous frame
   Frame prev_f;

   // render the point cloud in a separate thread
   std::thread render_loop;

   // Map consisting of all the points
   Map *mapp;

public:
   SLAM();
   SLAM(unsigned int H, unsigned int W, cv::Mat K);
   ~SLAM();

   void process(cv::Mat &img);
};

} // namespace dr3

#endif
