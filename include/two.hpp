#ifndef _TWO_VIEW_HPP_
#define _TWO_VIEW_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <string>

#include "utils.hpp"
#include "constants.hpp"

namespace dr3 {

class TwoView {
private:
   std::string _img_l;
   std::string _img_r;

   // Fundamental Matrix
   cv::Mat _F;
 
    // Essential Matrix
   cv::Mat _E;

   // camera calibration matrix
   cv::Mat _K;

   // Epipoles of the images
   cv::Mat epipole_l;
   cv::Mat epipole_r;
 
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
    
   // Estimate epipolar line for a given point
   cv::Point3d estimate_l(cv::Point2d pt, bool left=true); 
  
   // Given the Fundamental matrix, count the number of inliers 
   std::vector<int> get_inliers(cv::Mat F,
                                std::vector<cv::KeyPoint> kps_l,
                                std::vector<cv::KeyPoint> kps_r,
                                std::vector<cv::DMatch> matches);

};

} // namespace 3dr

#endif
