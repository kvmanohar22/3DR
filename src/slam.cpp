#include "slam.hpp"

namespace dr3 {

long unsigned int SLAM::cidx = 0;

SLAM::SLAM() {}

SLAM::SLAM(unsigned int H, 
           unsigned int W,
           cv::Mat K)
   : H(H), W(W), K(K) {

   v2d = new Viewer2D();
   v3d = new Viewer3D();

   I3x4 = cv::Mat::eye(cv::Size(4, 4), CV_32F);
   I3x4 = I3x4.rowRange(0, 3).colRange(0, 4);
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
   cv::Mat R1, t1, R2, t2;
   TwoView::extract_params(F, K, R1, R2, t1, t2);

   // Triangulation
   // 4 solutions are possible (pick the correct one)
   const size_t n_points = inliers.size();
   cv::Mat camera_p_pts(1, n_points, CV_32FC2);
   cv::Mat camera_c_pts(1, n_points, CV_32FC2);
   for (size_t i = 0; i < n_points; ++i) {
      cv::DMatch match = matches[inliers[i]];
      cv::Point2f pt_p = kps_p[match.queryIdx].pt;
      cv::Point2f pt_c = kps_c[match.trainIdx].pt;

      camera_p_pts.at<cv::Vec2f>(0, i) = {pt_p.x, pt_p.y};
      camera_c_pts.at<cv::Vec2f>(0, i) = {pt_c.x, pt_c.y};
   }

   // 4 possible solutions
   std::vector<cv::Mat> Rt(4, cv::Mat(cv::Size(4, 3), CV_32F));
   R1.copyTo(Rt[0].rowRange(0, 3).colRange(0, 3));
   t1.copyTo(Rt[0].rowRange(0, 3).col(3));
   R1.copyTo(Rt[1].rowRange(0, 3).colRange(0, 3));
   t2.copyTo(Rt[1].rowRange(0, 3).col(3));
   R2.copyTo(Rt[2].rowRange(0, 3).colRange(0, 3));
   t1.copyTo(Rt[2].rowRange(0, 3).col(3));
   R2.copyTo(Rt[3].rowRange(0, 3).colRange(0, 3));
   t2.copyTo(Rt[3].rowRange(0, 3).col(3));

   cv::Mat fpts4d;
   int max_z = -1;
   int idx = -1;

   // Update the camera poses
   // cv::Mat cam1 = K * prev_f.get_pose();
   for (int i = 0; i < 4; ++i) {
      // Set cameras
      cv::Mat cam1 = K * I3x4;
      cv::Mat Rt4x4 = cv::Mat::eye(cv::Size(4, 4), CV_32F);
      Rt[i].copySize(Rt4x4.rowRange(0, 3).colRange(0, 4));
      cv::Mat cam2 = K * Rt[i];
      // cv::Mat cam2 = K * prev_f.get_pose(false) * Rt4x4;

      // triangulate the points
      cv::Mat pts4d(1, n_points, CV_32FC4);
      // std::cout << pts4d.rows << " "
      //           << pts4d.cols << " "
      //           << std::endl;
      cv::triangulatePoints(cam1, cam2, camera_p_pts, camera_c_pts, pts4d);
      // std::cout << pts4d.rows << " "
      //           << pts4d.cols << " "
      //           << std::endl;
      // while(1);
      cv::Mat mask;
      utils::remove_nans(pts4d);
      int cz = 0;
      for (int j = 0; j < n_points; ++j) {
         float x = pts4d.at<float>(0, j);
         float y = pts4d.at<float>(1, j);
         float z = pts4d.at<float>(2, j);
         float w = pts4d.at<float>(3, j);
         if (z / w > 0)
            ++cz;
      }
      if (cz > max_z) {
         max_z = cz;
         idx = i;
         fpts4d = pts4d;
      }
      std::cout << ">>>> " << cz << "/" << n_points << std::endl;
   }

   // while (1);
   // update the camera poses
   if (idx == -1) {
      std::cerr << "WTF?" << std::endl;
      exit(1);
   }

   // update the correct orientation of the camera
   cv::Mat Rt4x4 = cv::Mat::eye(cv::Size(4, 4), CV_32F);
   Rt[idx].copySize(Rt4x4.rowRange(0, 3).colRange(0, 4));
   curr_f.set_pose(prev_f.get_pose(false) * Rt4x4);

   // Transfer the points to world frame
   for (int i = 0; i < n_points; ++i) {
      float x = fpts4d.at<float>(0, i);
      float y = fpts4d.at<float>(1, i);
      float z = fpts4d.at<float>(2, i);
      float w = fpts4d.at<float>(3, i);

      std::cout << i << ":  "
                << x / w << " "
                << y / w << " "
                << z / w << " "
                << std::endl;
      // cv::Mat pt_new = prev_f.get_pose(false) * cv::Mat(pt_old);
      // std::cout << pt_new << std::endl;

      // for (int k = 0; k < 4; ++k)
         // fpts4d.at<float>(i, k) = pt_new.at<float>(k);
   }

   while(1);
   // update the viewers
   _pts.push_back(fpts4d);
   v2d->update(img, curr_f.get_kps());
   v3d->update(Rt[idx], _pts);

   // Spit some debug data
   std::cout << "Processing frame: #" << std::setw(3) << cidx << " | "
             << "Matches: " << std::setw(3) << matches.size() << " | "
             << "Inliers: " << std::setw(3) << inliers.size() << " | "
             << "Triangulated: " << std::setw(3) << fpts4d.cols << " | "
             << "Camera t: " << std::setw(3) << prev_f.get_pose(false).col(3).t() << " | "
             << std::endl;

   // update
   prev_f = Frame(curr_f);
   ++cidx;
}

} // namespace dr3
