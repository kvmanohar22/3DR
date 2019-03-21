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
      prev_f = curr_f;
      mapp->add_frame(curr_f);
      ++cidx;
      return;
   }

   // Match the features
   std::vector<cv::KeyPoint> kps_p = prev_f->get_kps();
   std::vector<cv::KeyPoint> kps_c = curr_f->get_kps();
   cv::Mat des_p = prev_f->get_des();
   cv::Mat des_c = curr_f->get_des();
   std::vector<unsigned int> inliers;
   std::vector<cv::DMatch> matches;
   cv::BFMatcher bf = cv::BFMatcher(cv::NORM_HAMMING, true); 
   bf.match(des_p, des_c, matches);

   // Estimate Fundamental matrix
   cv::Mat F = TwoView::estimate_F(kps_p, kps_c, des_p, des_c, matches, inliers);

   // Recover (R, t) w.r.t previous frame (4 solutions)
   std::vector<cv::Mat> Rset, tset;
   TwoView::extract_camera_pose(F, K, Rset, tset);

   // Generate Xset for each (R, t)
   std::vector<std::vector<cv::Mat>> Xset;
   for (int i = 0; i < 4; ++i) {
      std::vector<cv::Mat> Xs;
      cv::Mat P1 = K * I3x4;
      cv::Mat temp(cv::Size(4, 3), CV_32F);
      Rset[i].copyTo(temp.rowRange(0, 3).colRange(0, 3));
      tset[i].copyTo(temp.rowRange(0, 3).col(3));
      cv::Mat P2 = K * temp;
      for (int ii = 0; ii < inliers.size(); ++ii) {
         cv::Mat xyz(cv::Size(1, 4), CV_32F);
         int qmatch = matches[inliers[ii]].queryIdx;
         int tmatch = matches[inliers[ii]].trainIdx;
         TwoView::triangulate(kps_p[qmatch],
                              kps_c[tmatch],
                              P1, P2, xyz);
         Xs.push_back(xyz);
      }
      Xset.push_back(Xs);
   }

   // Recover the correct camera pose
   cv::Mat R(cv::Size(3, 3), CV_32F);
   cv::Mat t(cv::Size(1, 3), CV_32F);
   std::vector<cv::Mat> X;
   std::vector<bool> inliers3d;
   int set_idx;
   TwoView::disambiguate_camera_pose(tset, Rset, Xset, t, R, inliers3d, set_idx);
   cv::Mat Rt4x4 = cv::Mat::eye(4, 4, CV_32F);
   R.copyTo(Rt4x4.rowRange(0, 3).colRange(0, 3));
   t.copyTo(Rt4x4.rowRange(0, 3).col(3));

   // set the pose of the current camera (world -> camera)
   curr_f->set_pose(Rt4x4 * prev_f->get_pose_w2c());

   // Transfer the points to world frame
   cv::Mat CtoW = prev_f->get_pose_c2w();
   cv::Mat Rt4x4inv = Rt4x4.inv();
   for (size_t ii = 0; ii < inliers3d.size(); ++ii) {
      if (!inliers3d[ii])
         continue;

      cv::Mat pt_new = CtoW * Rt4x4inv * Xset[set_idx][ii];
      cv::Mat xyz = pt_new.rowRange(0, 3) / pt_new.at<float>(3);

      // Register a new point
      Point *point = new Point(xyz, mapp->n_points());

      // Add observation of frame in point
      point->add_observation(curr_f, matches[inliers[ii]].queryIdx);
      point->add_observation(prev_f, matches[inliers[ii]].trainIdx);

      // Add observation of point in frame
      curr_f->add_observation(point, matches[inliers[ii]].queryIdx);
      prev_f->add_observation(point, matches[inliers[ii]].trainIdx);

      // Add the point to the map
      mapp->add_point(point);
   }
   // Register the new camera with map
   mapp->add_frame(curr_f);

   // update the viewers
   v2d->update(img, curr_f->get_kps());

   // Spit some debug data
   std::cout << "Processing frame: #" << std::setw(3) << cidx << " | "
             << "Matches: " << std::setw(3) << matches.size() << " | "
             << "Inliers: " << std::setw(3) << inliers.size() << " | "
             << "#KeyFrames: " << std::setw(3) << mapp->n_frames() << " | "
             << "#points: " << std::setw(7) << mapp->n_points() << " | "
             << "Camera pos: " << std::setw(3) << curr_f->get_center().t()
             << std::endl;

   // update the previous frame
   prev_f = curr_f;
   ++cidx;
}

} // namespace dr3
