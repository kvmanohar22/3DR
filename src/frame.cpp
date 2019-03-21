#include "frame.hpp"

namespace dr3 {

Frame::Frame() {}

Frame::Frame(Frame &frame)
   : idx(frame.get_idx()), kps(frame.get_kps()),
     des(frame.get_des()), pose_w2c(frame.get_pose(false)),
     pose_c2w(frame.get_center(false)) {}

Frame::Frame(const long unsigned int idx,
             const cv::Mat &img, const cv::Mat &K) 
   : idx(idx), K(K) {

   compute_kps(img);
   if (idx == 0) {
      pose_w2c = cv::Mat::eye(cv::Size(4, 4), CV_32F);
      update_poses();
   } else {

   }
}

void Frame::compute_kps(const cv::Mat &img) {
   cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
   orb->detectAndCompute(img, cv::noArray(), kps, des);
}

void Frame::set_pose(cv::Mat pose) {
   this->pose_w2c = pose;
   update_poses();
}

void Frame::update_poses() {
   pose_c2w = pose_w2c.inv();
}

cv::Mat Frame::get_pose(bool _short) const {
   if (_short) {
      return pose_w2c.rowRange(0, 3).colRange(0, 4).clone();
   } else {
      return pose_w2c.clone();
   }
}

cv::Mat Frame::get_center(bool _short) const {
   if (_short) {
      return pose_c2w.rowRange(0, 3).colRange(0, 4).clone();
   } else {
      return pose_c2w.clone();
   }
}

} // namespace dr3
