#ifndef _VIEWER_HPP_
#define _VIEWER_HPP_

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pangolin/pangolin.h>

#include <iostream>

#include "utils.hpp"

namespace dr3 {

// 2D Viewer
class Viewer2D {
public:
   Viewer2D();

   // update the view
   cv::Mat update(cv::Mat img_l, cv::Mat img_r,
                  std::vector<cv::KeyPoint> kps_l,
                  std::vector<unsigned int> idxs_l,
                  std::vector<cv::KeyPoint> kps_r,
                  std::vector<unsigned int> idxs_r);

   void draw_kps(cv::Mat &img,
                 std::vector<cv::KeyPoint> kps_l,
                 std::vector<unsigned int> idxs_l,
                 std::vector<cv::KeyPoint> kps_r,
                 std::vector<unsigned int> idxs_r);

   void draw_point(cv::Mat &img,
                   cv::KeyPoint pt,
                   cv::Scalar color = cv::Scalar(0, 255, 0));

   void draw_point(cv::Mat &img,
                   cv::Point2f pt,
                   cv::Scalar color = cv::Scalar(0, 255, 0));

   // The line should be normalized (a**2+b**2 = 1)
   void draw_line(cv::Mat &img,
                  cv::Point3f line,
                  cv::Scalar color = cv::Scalar(0, 255, 0));

}; // class Viewer2D

// 3D Viewer
class Viewer3D {
private:
   size_t H, W;
   pangolin::View d_cam;
   pangolin::OpenGlRenderState s_cam;

public:
   Viewer3D();
   Viewer3D(size_t H, size_t W);

   // Initialize the viewer
   void init();

   // update after each frame
   void update();
}; // class Viewer3D


} // namespace dr3

#endif
