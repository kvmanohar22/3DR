#ifndef _TWO_VIEW_HPP_
#define _TWO_VIEW_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <string>

#include "utils.hpp"
#include "constants.hpp"
#include "viewer.hpp"

namespace dr3 {

class TwoView {
private:
   std::string _img_l;
   std::string _img_r;

   // Images
   cv::Mat img_l;
   cv::Mat img_r;

   // Fundamental Matrix
   cv::Mat _F;
 
    // Essential Matrix
   cv::Mat _E;

   // camera calibration matrix
   cv::Mat _K;

   // Epipoles of the images
   cv::Mat epipole_l;
   cv::Mat epipole_r;

   // keypoints
   std::vector<cv::KeyPoint> kps_l, kps_r;
   std::vector<cv::DMatch> matches;
   std::vector<unsigned int> inliers;
 
public:
   TwoView();
   TwoView(std::string img_l, std::string img_r);

   // Estimate Fundamental/Essential matrix
   cv::Mat estimate_F();
   cv::Mat estimate_E();

   // Reduce the estimated Fundamental matrix to a rank 2 matrix
   cv::Mat clean_F(cv::Mat F);
 
   // Estimate epipoles of the images
   void estimate_epipoles();

   // Draw some points and lines
   cv::Mat draw_poles_and_lines(size_t n=20, bool left_points=true);

   // Estimate epipolar line for a given point
   cv::Point3d estimate_l(cv::Point2d pt, bool left=true); 
  
   // Given the Fundamental matrix, count the number of inliers 
   std::vector<unsigned int> get_inliers(cv::Mat F,
                                std::vector<cv::KeyPoint> kps_l,
                                std::vector<cv::KeyPoint> kps_r,
                                std::vector<cv::DMatch> matches);

};

} // namespace 3dr

#endif
