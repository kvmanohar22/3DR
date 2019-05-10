#include "slam.hpp"
#include "camera.hpp"

namespace dr3 {

long unsigned int SLAM::cidx = 0;

SLAM::SLAM() {}

SLAM::SLAM(size_t H, 
           size_t W,
           cv::Mat K, int argc, char **argv)
   : H(H), W(W), K(K) {

   // Map to store the point cloud
   mapp = new Map();

   // Viewers
   v2d = new Viewer2D();
   v3d = new Viewer3D(mapp);

   cam = new Pinhole(50, 50, 50, 50, 50, 50);

   // Identity
   I3x4 = cv::Mat::eye(cv::Size(4, 4), CV_32F);
   I3x4 = I3x4.rowRange(0, 3).colRange(0, 4);

   // start the render thread
   render_loop = std::thread(&Viewer3D::update, v3d);

   // Monitor
   monitor = new Monitor(); 

   monitor->add_timer("global");        // Global timing monitor
   monitor->add_timer("frame");         // Generating frame object for each image
   monitor->add_timer("match");         // Matching keypoints across two frames
   monitor->add_timer("fmatrix");       // Estimating Fundamental matrix
   monitor->add_timer("triangulation"); // Estimating Rset, tset, Xset & the correct points
   monitor->add_timer("optimizer");     // Bundle Adjustment
}

SLAM::~SLAM() {
   delete v2d;
   delete v3d;
   delete mapp;
   render_loop.join();
}

void SLAM::pprint(Frame &curr_f) {
   const cv::Mat pose = curr_f.get_center().t();
   const double fps = mapp->n_frames()  / monitor->get_ct("global");
   const size_t n_frames = mapp->n_frames();
   const size_t n_points = mapp->n_points();
   const size_t n_params = n_frames * 6 + n_points * 3;
  
   std::cout << std::right << std::setw(70) << "Frame Number" << " : "
             << std::left << std::setw(20) << cidx << std::endl
             << std::right << std::setw(70) << "FPS" << " : "
             << std::left << std::setw(20) << fps << std::endl
             << std::right << std::setw(70) << "Total nodes (F + P)" << " : "
             << std::right << std::setw(4) << n_frames << " + " << std::setw(5) << n_points << " = "
             << std::setw(5) << n_frames + n_points << std::endl
             << std::right << std::setw(70) << "Total Params" << " : "
             << std::left << std::setw(20) << n_params << std::endl
             << std::right << std::setw(70) << "Total observations" << " : "
             << std::left << std::setw(20) << mapp->n_observations() << std::endl
             << std::right << std::setw(70) << "Current Camera pos" << " : "
             << std::left << pose << std::endl
             << std::setw(65) << ' ' << "-------------" << std::endl 
             << std::right << std::setw(70) << "frame generation" << " : "
             << std::left << std::setw(20) << monitor->get_at("frame") << std::endl
             << std::right << std::setw(70) << "KeyPoint matching" << " : "
             << std::left << std::setw(20) << monitor->get_at("match") << std::endl
             << std::right << std::setw(70) << "F estimation" << " : "
             << std::left << std::setw(20) << monitor->get_at("fmatrix") << std::endl
             << std::right << std::setw(70) << "Triangulation (Rset, tset, Xset)" << " : "
             << std::left << std::setw(20) << monitor->get_at("triangulation") << std::endl
             << std::right << std::setw(70) << "Optimizer" << " : "
             << std::left << std::setw(20) << monitor->get_at("optimizer") << std::endl
             << std::setw(65) << ' ' << "-------------" << std::endl 
             << std::right << std::setw(70) << "Total" << " : "
             << std::left << std::setw(20) << monitor->get_tat() << std::endl
             << std::endl;
}

void SLAM::process(cv::Mat &img) {
   std::cout << "here " << std::endl;
   monitor->tic("global");

   // process the current frame
   monitor->tic("frame");
   Frame *curr_f = new Frame(cidx, img, cam);
   monitor->toc("frame");

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
   
   monitor->tic("match");
   bf.match(des_p, des_c, matches);
   monitor->toc("match");

   // Estimate Fundamental matrix
   monitor->tic("fmatrix");
   cv::Mat F = TwoView::estimate_F(kps_p, kps_c, des_p, des_c, matches, inliers);
   monitor->toc("fmatrix");
   
   // Recover (R, t) w.r.t previous frame (4 solutions)
   std::vector<cv::Mat> Rset, tset;
   monitor->tic("triangulation");
   TwoView::extract_camera_pose(F, K, Rset, tset);

   // Generate Xset for each (R, t)
   std::vector<std::vector<cv::Mat>> Xset(4);
   for (int i = 0; i < 4; ++i) {
      std::vector<cv::Mat> Xs(inliers.size());
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
         Xs[ii] = xyz;
      }
      Xset[i] = Xs;
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
   monitor->toc("triangulation");
   
   // set the pose of the current camera (world -> camera)
   curr_f->set_pose(Rt4x4 * prev_f->get_pose_w2c());

   // Transfer the points to world frame
   cv::Mat CtoW = prev_f->get_pose_c2w();
   cv::Mat Rt4x4inv = Rt4x4.inv();
   int duplicate_count = 0;
   for (size_t ii = 0; ii < inliers3d.size(); ++ii) {
      if (!inliers3d[ii])
         continue;

      // Avoid adding duplicate points (only considering previous frame)
      size_t qidx = matches[inliers[ii]].queryIdx;
      size_t tidx = matches[inliers[ii]].trainIdx;
      if (!prev_f->fresh_point(qidx)) {
         Point *point = prev_f->get_point(qidx);
         curr_f->add_observation(point, tidx);
         point->add_observation(curr_f, tidx);
         ++duplicate_count;
         continue;
      }

      // This is a new point not observed in any of the previous frames
      cv::Mat pt_new = CtoW * Rt4x4inv * Xset[set_idx][ii];
      cv::Mat xyz = pt_new.rowRange(0, 3) / pt_new.at<float>(3);

      // Register a new point
      Point *point = new Point(xyz, mapp->n_points());

      // Add observation of frame in point
      point->add_observation(prev_f, qidx);
      point->add_observation(curr_f, tidx);

      // Add observation of point in frame
      prev_f->add_observation(point, qidx);
      curr_f->add_observation(point, tidx);

      // Add the point to the map
      mapp->add_point(point);
   }
   // Register the new camera with map
   mapp->add_frame(curr_f);

   // update the viewers
   v2d->update(img, curr_f->get_kps());

   // Bundle Adjustment
   monitor->tic("optimizer");
   // Optimizer::global_BA(mapp, K);
   monitor->toc("optimizer");

   // update the previous frame
   prev_f = curr_f;
   ++cidx;
   monitor->toc("global");

   // Spit some debug data
   pprint(*curr_f);
}

} // namespace dr3
