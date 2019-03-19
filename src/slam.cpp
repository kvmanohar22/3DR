#include "slam.hpp"

namespace dr3 {

long unsigned int SLAM::cidx = 0;

SLAM::SLAM() {}

SLAM::SLAM(unsigned int H, 
           unsigned int W,
           cv::Mat K)
   : H(H), W(W), K(K) {

   // Map to store the point cloud
   mapp = new Map();

   // Viewers
   v2d = new Viewer2D();
   v3d = new Viewer3D(mapp);

   // Identity
   I3x4 = cv::Mat::eye(cv::Size(4, 4), CV_32F);
   I3x4 = I3x4.rowRange(0, 3).colRange(0, 4);

   // start the render thread
   render_loop = std::thread(&Viewer3D::update, v3d);
}

SLAM::~SLAM() {
   delete v2d;
   delete v3d;
   delete mapp;
   render_loop.join();
}

void SLAM::process(cv::Mat &img) {
   // process the current frame
   Frame *curr_f = new Frame(cidx, img, K);
   if (cidx == 0) {
      prev_f = Frame(*curr_f);
      ++cidx;
      return;
   }

   // Match the features from previous frame and current frame
   std::vector<cv::KeyPoint> kps_p = prev_f.get_kps();
   std::vector<cv::KeyPoint> kps_c = curr_f->get_kps();
   cv::Mat des_p = prev_f.get_des();
   cv::Mat des_c = curr_f->get_des();
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

   cv::Mat fpts4d;
   int max_z = -1;
   int idx = -1;

   // Update the camera poses
   for (int i = 0; i < 4; ++i) {
      // Set cameras
      cv::Mat Rtmat(cv::Size(4, 3), CV_32F);
      switch(i) {
         case 0:
            R1.copyTo(Rtmat.rowRange(0, 3).colRange(0, 3));
            t1.copyTo(Rtmat.col(3));
            break;
         case 1:
            R1.copyTo(Rtmat.rowRange(0, 3).colRange(0, 3));
            t2.copyTo(Rtmat.col(3));
            break;
         case 2:
            R2.copyTo(Rtmat.rowRange(0, 3).colRange(0, 3));
            t1.copyTo(Rtmat.col(3));
            break;
         case 3:
            R2.copyTo(Rtmat.rowRange(0, 3).colRange(0, 3));
            t2.copyTo(Rtmat.col(3));
            break;
      }

      cv::Mat cam1 = K * prev_f.get_pose(true);
      cv::Mat cam2 = K * Rtmat * prev_f.get_pose(false);

      cv::Mat tvec = Rtmat.col(3);

      /*
         cam.z > 0 => the camera is moving in -ve z direction
                   => the points should have -ve z in world frame
      */

      // triangulate the points
      cv::Mat pts4d(1, n_points, CV_32FC4);
      cv::triangulatePoints(cam1, cam2, camera_p_pts, camera_c_pts, pts4d);

      // utils::remove_nans(pts4d);
      int cz = 0;
      for (int j = 0; j < n_points; ++j) {
         float z = pts4d.at<float>(2, j);
         float w = pts4d.at<float>(3, j);
         if (tvec.at<float>(2) > 0) {
            if (z / w < 0)
            ++cz;
         } else {
            if (z / w > 0)
            ++cz;
         }
      }
      if (cz > max_z) {
         max_z = cz;
         idx = i;
         fpts4d = pts4d;
      }
   }

   // update the camera poses
   if (idx == -1) {
      std::cerr << "WTF?" << std::endl;
      exit(1);
   }

   // update the correct orientation of the camera
   cv::Mat Rt4x4 = cv::Mat::eye(cv::Size(4, 4), CV_32F);
   switch(idx) {
      case 0:
         R1.copyTo(Rt4x4.rowRange(0, 3).colRange(0, 3));
         t1.copyTo(Rt4x4.rowRange(0, 3).col(3));
         break;
      case 1:
         R1.copyTo(Rt4x4.rowRange(0, 3).colRange(0, 3));
         t2.copyTo(Rt4x4.rowRange(0, 3).col(3));
         break;
      case 2:
         R2.copyTo(Rt4x4.rowRange(0, 3).colRange(0, 3));
         t1.copyTo(Rt4x4.rowRange(0, 3).col(3));
         break;
      case 3:
         R2.copyTo(Rt4x4.rowRange(0, 3).colRange(0, 3));
         t2.copyTo(Rt4x4.rowRange(0, 3).col(3));
         break;
   }

   // set the pose of the current camera (world -> camera)
   curr_f->set_pose(prev_f.get_pose(false) * Rt4x4);

   // Transfer the points to world frame
   cv::Mat CtoW = prev_f.get_pose(false).inv();
   for (int i = 0; i < n_points; ++i) {
      float x = fpts4d.at<float>(0, i);
      float y = fpts4d.at<float>(1, i);
      float z = fpts4d.at<float>(2, i);
      float w = fpts4d.at<float>(3, i);

      cv::Mat pt_old = (cv::Mat_<float>(4, 1) << x, y, z, w);
      cv::Mat pt_new = CtoW * pt_old;

      w = pt_new.at<float>(3);
      cv::Mat xyz(cv::Size(3, 1), CV_32F);
      for (int k = 0; k < 3; ++k)
         xyz.at<float>(0, k) = pt_new.at<float>(k) / w;

      // Register a new point
      Point *point = new Point(xyz);
      point->add_observation(curr_f, 0);

      // Add the point to the map
      mapp->add_point(point);
   }
   // Register the new camera with map
   mapp->add_frame(curr_f);

   // update the viewers
   v2d->update(img, curr_f->get_kps());
   // v3d->update(Rt4x4, _pts);

   // Spit some debug data
   std::cout << "Processing frame: #" << std::setw(3) << cidx << " | "
             << "Matches: " << std::setw(3) << matches.size() << " | "
             << "Inliers: " << std::setw(3) << inliers.size() << " | "
             << "Triangulated: " << std::setw(3) << fpts4d.cols << " | "
             << "#frames: " << std::setw(3) << mapp->n_frames() << " | "
             << "#points: " << std::setw(3) << mapp->n_points() << " | "
             << "Camera t: " << std::setw(3) << prev_f.get_pose(false).col(3).t()
             << std::endl;

   // update
   prev_f = Frame(*curr_f);
   ++cidx;
}

} // namespace dr3
