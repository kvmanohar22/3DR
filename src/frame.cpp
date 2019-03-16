#include "frame.hpp"

namespace dr3 {

Frame::Frame() {}

Frame::Frame(Frame &frame)
   : idx(frame.get_idx()), kps(frame.get_kps()),
     des(frame.get_des()), pose(frame.get_pose()) {}

Frame::Frame(const long unsigned int idx,
             const cv::Mat &img, const cv::Mat &K) 
   : idx(idx), K(K) {

   compute_kps(img);
   if (idx == 0) {
      pose = cv::Mat::eye(4, 4, CV_32F);
   } else {

   }
}

void Frame::compute_kps(const cv::Mat &img) {
   cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
   orb->detectAndCompute(img, cv::noArray(), kps, des);
}

} // namespace dr3
