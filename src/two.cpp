#include "two.hpp"

namespace dr3 {

TwoView::TwoView() { 
  _img_l = "";
  _img_r = "";
}

TwoView::TwoView(std::string _img_l, std::string _img_r) { 
  this->_img_l = _img_l;
  this->_img_r = _img_r;
}

cv::Mat TwoView::estimate_F() {
   cv::Mat img_l, gray_l;
   cv::Mat img_r, gray_r;
  
   img_l = utils::load_image(_img_l); 
   img_r = utils::load_image(_img_r); 

   cv::cvtColor(img_l, gray_l, cv::COLOR_BGR2GRAY); 
   cv::cvtColor(img_r, gray_r, cv::COLOR_BGR2GRAY); 

   cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
   
   std::vector<cv::KeyPoint> left_kps, right_kps;
   cv::Mat left_des, right_des;
   orb->detectAndCompute(gray_l, cv::noArray(), left_kps, left_des);
   orb->detectAndCompute(gray_r, cv::noArray(), right_kps, right_des);
   
   std::vector<cv::DMatch> matches;
   cv::BFMatcher bf = cv::BFMatcher(cv::NORM_HAMMING, true); 
   bf.match(left_des, right_des, matches);

   cv::Mat F;
   const int N = 8; // 8-point algorithm
   const int max_index = matches.size(); 
   for (int i = 0; i < RANSAC_ITERS; ++i) {
      std::set<int> indices;
      std::vector<cv::DMatch> match_subset;
      while (indices.size() < std::min(N, max_index)) {
         int rand_idx = rand() % max_index;
         if (indices.find(rand_idx) == indices.end()) {
            match_subset.push_back(matches[rand_idx]);
            indices.insert(rand_idx); 
         } 
      }

      cv::Mat A = cv::Mat::zeros(cv::Size(9, 8), CV_32F);
      for (int j = 0; j < N; ++j) {
         cv::DMatch match = match_subset[j]; 
         int q_idx = match.queryIdx; 
         int t_idx = match.trainIdx; 
   
         float ul, vl, ur, vr;
         ul = left_kps[q_idx].pt.x;
         vl = left_kps[q_idx].pt.y;
         ur = right_kps[t_idx].pt.x;
         vr = right_kps[t_idx].pt.y;
         
         float row_data[] = {
            ur*ul, ur*vl, ur*ur,
            vr*ul, vr*vl, vr*vr,
            ul, vl, 1}; 
         cv::Mat row = cv::Mat(cv::Size(1, 9), CV_32F, &row_data);
         row.copyTo(A.row(j));
      }

      // SVD
      cv::Mat F_est = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
      cv::SVD svd(A, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
      F_est = svd.vt.row(svd.vt.rows-1).reshape(3, 3); 

      std::vector<int> inliers = get_inliers(F_est);
   }
   return F.clone();
}

cv::Mat TwoView::estimate_E() {
   cv::Mat E;
   return E.clone();
}

void TwoView::estimate_epipoles() {

}

cv::Point3d TwoView::estimate_l(cv::Point2d pt, bool left) {
   cv::Point3d line;

   return line;
}

std::vector<int> TwoView::get_inliers(cv::Mat F) {
   std::vector<int> inliers;

   return inliers;
}

} // namespace 3dr

