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
      mapp->add_frame(curr_f);
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

   std::vector<cv::Point2f> pts1(matches.size());
   std::vector<cv::Point2f> pts2(matches.size());

   for (int i = 0; i < matches.size(); ++i) {
      cv::Point2f pt1, pt2;
      int qmatch = matches[i].queryIdx;
      int tmatch = matches[i].trainIdx;
      pt1.x = kps_p[qmatch].pt.x;
      pt1.y = kps_p[qmatch].pt.y;
      pt2.x = kps_c[tmatch].pt.x;
      pt2.y = kps_c[tmatch].pt.y;
      pts1.push_back(pt1);
      pts2.push_back(pt2);
   }
   // estimate Fundamental matrix
   cv::Mat fest = cv::findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 3, 0.99);
   cv::Mat F = TwoView::estimate_F(kps_p, kps_c, des_p, des_c, matches, inliers);
   F.convertTo(F, CV_32F);

   // Recover (R, t) w.r.t previous frame
   std::vector<cv::Mat> Rset, tset;
   TwoView::extract_camera_pose(F, K, Rset, tset);

   // Generate Xset for each (R, t)
   std::vector<std::vector<cv::Mat>> Xset;
   for (int i = 0; i < 4; ++i) {
      std::vector<cv::Mat> Xs;
      cv::Mat P1 = K * prev_f.get_pose(true);
      cv::Mat temp(cv::Size(4, 3), CV_32F);
      Rset[i].copyTo(temp.rowRange(0, 3).colRange(0, 3));
      tset[i].copyTo(temp.rowRange(0, 3).col(3));
      cv::Mat P2 = K * temp * prev_f.get_pose(false);
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

   cv::Mat R(cv::Size(3, 3), CV_32F);
   cv::Mat t(cv::Size(1, 3), CV_32F);
   std::vector<cv::Mat> X;
   TwoView::disambiguate_camera_pose(tset, Rset, Xset, t, R, X);
   cv::Mat Rt4x4 = cv::Mat::eye(4, 4, CV_32F);
   R.copyTo(Rt4x4.rowRange(0, 3).colRange(0, 3));
   t.copyTo(Rt4x4.rowRange(0, 3).col(3));

   // set the pose of the current camera (world -> camera)
   curr_f->set_pose(Rt4x4 * prev_f.get_pose(false));

   // Transfer the points to world frame
   cv::Mat CtoW = prev_f.get_center(false);
   for (int i = 0; i < X.size(); ++i) {
      float x = X[i].at<float>(0);
      float y = X[i].at<float>(1);
      float z = X[i].at<float>(2);
      float w = X[i].at<float>(3);

      cv::Mat pt_old = (cv::Mat_<float>(4, 1) << x, y, z, w);
      cv::Mat pt_new = CtoW * pt_old;

      w = pt_new.at<float>(3);
      cv::Mat xyz(cv::Size(1, 3), CV_32F);
      for (int k = 0; k < 3; ++k)
         xyz.at<float>(k) = pt_new.at<float>(k) / w;

      // Register a new point
      Point *point = new Point(xyz);
      point->add_observation(curr_f, 0);

      // Add the point to the map
      mapp->add_point(point);
   }
   // Register the new camera with map
   mapp->add_frame(curr_f);

   // update the viewers
   // v2d->update(img, curr_f->get_kps());

   // Spit some debug data
   std::cout << "Processing frame: #" << std::setw(3) << cidx << " | "
             << "Matches: " << std::setw(3) << matches.size() << " | "
             << "Inliers: " << std::setw(3) << inliers.size() << " | "
             << "#frames: " << std::setw(3) << mapp->n_frames() << " | "
             << "#points: " << std::setw(7) << mapp->n_points() << " | "
             << "Camera C: " << std::setw(3) << curr_f->get_camc().t()
             << std::endl;

   // update the previous frame
   prev_f = Frame(*curr_f);
   ++cidx;

   if (cidx == 2)
      while(1);
}

} // namespace dr3
