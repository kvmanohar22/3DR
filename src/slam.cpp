#include "slam.hpp"

namespace dr3 {

long unsigned int SLAM::cidx = 0;

SLAM::SLAM() {}

SLAM::SLAM(unsigned int H, 
           unsigned int W,
           cv::Mat K)
   : H(H), W(W), K(K) {

   v2d = new Viewer2D();
   // v3d = new Viewer3D();
}

void SLAM::process(cv::Mat &img) {
   // process the current frame
   Frame curr_f(cidx, img, K);
   if (cidx == 0) {
      prev_f = Frame(curr_f);
      ++cidx;
      return;
   }

   // Match the features from previous frame and current frame
   std::vector<cv::KeyPoint> kps_p = prev_f.get_kps();
   std::vector<cv::KeyPoint> kps_c = curr_f.get_kps();
   cv::Mat des_p = prev_f.get_des();
   cv::Mat des_c = curr_f.get_des();
   std::vector<unsigned int> inliers;

   std::vector<cv::DMatch> matches;
   cv::BFMatcher bf = cv::BFMatcher(cv::NORM_HAMMING, true); 
   bf.match(des_p, des_c, matches);

   // estimate Fundamental matrix
   cv::Mat F = TwoView::estimate_F(kps_p, kps_c, des_p, des_c, matches, inliers);

   // Recover (R, t) w.r.t previous frame
   cv::Mat R, t;
   TwoView::extract_params(F, K, R, t);

   // update the viewers
   v2d->update(img, curr_f.get_kps());

   // Spit some debug data
   std::cout << "Processing frame: #" << std::setw(3) << cidx << " | "
             << "Matches: " << std::setw(3) << matches.size() << " | "
             << "Inliers: " << std::setw(3) << inliers.size() << " | "
             << std::endl;

   // update
   prev_f = Frame(curr_f);
   ++cidx;
}

} // namespace dr3
