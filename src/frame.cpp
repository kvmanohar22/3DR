#include "frame.hpp"

namespace dr3 {

Frame::Frame() {}

Frame::Frame(Frame &frame)
   : idx(frame.get_idx()), kps(frame.get_kps()),
     des(frame.get_des()), pose(frame.get_pose(false)) {}

Frame::Frame(const long unsigned int idx,
             const cv::Mat &img, const cv::Mat &K) 
   : idx(idx), K(K) {

   compute_kps(img);
   if (idx == 0) {
      pose = cv::Mat::eye(cv::Size(4, 4), CV_32F);
   } else {

   }
}

void Frame::compute_kps(const cv::Mat &img) {
   cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
   orb->detectAndCompute(img, cv::noArray(), kps, des);
}

void Frame::set_pose(cv::Mat pose) {
   this->pose = pose;
}

cv::Mat Frame::get_pose(bool _short) const {
   if (_short) {
      return pose.rowRange(0, 3).colRange(0, 4).clone();
   } else {
      return pose.clone();
   }
}

} // namespace dr3
