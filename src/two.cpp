#include "two.hpp"
#include <cstdlib>
#include <cmath>

namespace dr3 {
using namespace std;

TwoView::TwoView() { 
  _img_l = "";
  _img_r = "";
}

TwoView::TwoView(std::string _img_l, std::string _img_r) { 
  this->_img_l = _img_l;
  this->_img_r = _img_r;
}

cv::Mat TwoView::estimate_F() {
   cv::Mat gray_l, gray_r;

   img_l = utils::load_image(_img_l); 
   img_r = utils::load_image(_img_r); 

   cv::cvtColor(img_l, gray_l, cv::COLOR_BGR2GRAY); 
   cv::cvtColor(img_r, gray_r, cv::COLOR_BGR2GRAY); 

   cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
   
   std::vector<cv::KeyPoint> kps_l, kps_r;
   cv::Mat left_des, right_des;
   orb->detectAndCompute(gray_l, cv::noArray(), kps_l, left_des);
   orb->detectAndCompute(gray_r, cv::noArray(), kps_r, right_des);
   
   std::vector<cv::DMatch> matches;
   cv::BFMatcher bf = cv::BFMatcher(cv::NORM_HAMMING, true); 
   bf.match(left_des, right_des, matches);

   cv::Mat F;
   const int N = 8; // 8-point algorithm
   size_t max_inliers = 0;
   std::vector<unsigned int> best_inliers;
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
         ul = kps_l[q_idx].pt.x;
         vl = kps_l[q_idx].pt.y;
         ur = kps_r[t_idx].pt.x;
         vr = kps_r[t_idx].pt.y;

         float row_data[] = {
            ur*ul, ur*vl, ur,
            vr*ul, vr*vl, vr,
            ul, vl, 1};
         cv::Mat row = cv::Mat(cv::Size(9, 1), CV_32F, &row_data);
         row.copyTo(A.row(j));
      }

      // SVD
      cv::Mat F_est = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
      cv::SVD svd(A, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
      F_est = svd.vt.row(svd.vt.rows-1).reshape(1, 3); 

      F_est = this->clean_F(F_est);
      std::vector<unsigned int> inliers = get_inliers(F_est, kps_l, kps_r, matches);

      size_t n_inliers = inliers.size();
      if (n_inliers > max_inliers) {
         max_inliers = n_inliers;
         best_inliers = inliers;
         F = F_est;
      }

      #ifdef DEBUG
         std::cout << "RANSAC ITER: " << i+1 << "/" << RANSAC_ITERS << " " 
                   << "MAX INLIERS: " << max_inliers << " " 
                   << "CURR INLIERS: " << inliers.size() << " "
                   << "TOTAL MATCHES: "<< matches.size() << " "
                   << std::endl;
      #endif
   }

   this->kps_l = kps_l;
   this->kps_r = kps_r;
   this->matches = matches;
   this->inliers = best_inliers;
   this->_F = F;

   return F.clone();
}

cv::Mat TwoView::clean_F(cv::Mat F) {
   cv::SVD svd(F, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
   cv::Mat D = svd.w;

   float d1 = D.at<float>(0);
   float d2 = D.at<float>(1);

   float d_data[] = {d1, 0 , 0,
                     0 , d2, 0,
                     0 , 0 , 0};
   cv::Mat d_hat(cv::Size(3, 3), CV_32F, &d_data);

   cv::Mat _F = svd.u * d_hat * svd.vt;
   return _F.clone();
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

std::vector<unsigned int> TwoView::get_inliers(cv::Mat F,
                                      std::vector<cv::KeyPoint> kps_l,
                                      std::vector<cv::KeyPoint> kps_r,
                                      std::vector<cv::DMatch> matches) {
   std::vector<unsigned int> inliers;
   for (int i = 0; i < matches.size(); ++i) {
      int q_idx = matches[i].queryIdx;
      int t_idx = matches[i].trainIdx;

      cv::KeyPoint kpl = kps_l[q_idx];
      float x1_data[] = {kpl.pt.x, kpl.pt.y, 1};
      cv::Mat x1(cv::Size(1, 3), CV_32F, &x1_data);
      cv::KeyPoint kpr = kps_r[t_idx];
      float x2_data[] = {kpr.pt.x, kpr.pt.y, 1};
      cv::Mat x2(cv::Size(1, 3), CV_32F, &x2_data);

      float a = F.row(0).t().dot(x1);
      float b = F.row(1).t().dot(x1);

      float d = abs(x2.dot(F * x1)) / sqrt(a*a + b*b);

      if (d < RANSAC_THRESH)
         inliers.push_back(i);
   }

   return inliers;
}

cv::Mat TwoView::draw_poles_and_lines(size_t n, bool left_points) {
   n = std::min(n, inliers.size());
   Viewer2D v2d;
   cv::Mat timg_l = this->img_l;
   cv::Mat timg_r = this->img_r;
   for (int i = 0; i < n; ++i) {
      cv::KeyPoint kpl = kps_l[matches[inliers[i]].queryIdx];
      cv::KeyPoint kpr = kps_r[matches[inliers[i]].trainIdx];

      float data_l[] = {kpl.pt.x, kpl.pt.y, 1};
      cv::Mat xl(cv::Size(1, 3), CV_32F, &data_l);
      float data_r[] = {kpl.pt.x, kpl.pt.y, 1};
      cv::Mat xr(cv::Size(1, 3), CV_32F, &data_r);

      cv::Mat line;
      if (left_points)
         line = this->_F * xl;
      else
         line = this->_F.t() * xr;

      cv::Point3f pt3;
      float a = line.at<float>(0);
      float b = line.at<float>(1);
      float c = line.at<float>(2);
      float w = 1./std::sqrt(a*a + b*b);
      pt3.x = a*w;
      pt3.y = b*w;
      pt3.z = c*w;

      cv::Scalar color = utils::getc();
      v2d.draw_point(timg_l, kpl, color);
      v2d.draw_point(timg_r, kpr, color);
      if (left_points)
         v2d.draw_line(timg_r, pt3, color);
      else
         v2d.draw_line(timg_l, pt3, color);
   }

   cv::Mat img = v2d.update(timg_l, timg_r, kps_l, inliers, kps_r, inliers);
   return img;
}

} // namespace 3dr
