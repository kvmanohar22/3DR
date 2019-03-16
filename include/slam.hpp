#ifndef _SLAM_HPP_
#define _SLAM_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <iomanip>

#include "viewer.hpp"
#include "frame.hpp"
#include "two.hpp"

namespace dr3 {

class SLAM {
private:
   // Image dimensions
   unsigned int H, W;

   // camera intrinsics
   cv::Mat K;

   // viewers
   Viewer2D *v2d;
   Viewer3D *v3d;

   // ID of the current frame processed
   static long unsigned int cidx;

   // current and previous frames
   Frame prev_f, curr_f;

public:
   SLAM();
   SLAM(unsigned int H, unsigned int W, cv::Mat K);

   void process(cv::Mat &img);
};

} // namespace dr3

#endif
