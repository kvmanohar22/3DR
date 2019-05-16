#include <features.hpp>
#include "frame.hpp"

namespace dr3 {

Frame::Frame(Frame &frame)
   : idx(frame.get_idx()), kps(frame.get_kps()),
     des(frame.get_des()),
     pose_w2c(frame.get_pose_w2c()),
     pose_c2w(frame.get_pose_c2w()),
     center(frame.get_center()) {}

Frame::Frame(const long unsigned int idx,
             const cv::Mat &img, AbstractCamera *cam,
             double timestamp)
   : idx(idx), _cam(cam), _time_stamp(timestamp), _is_keyframe(false) {

    utils::create_img_pyramid(img, Config::n_pyr_levels(), _img_pyr);
    points = std::vector<Point*>(kps.size(), nullptr);
}

void Frame::compute_kps(const cv::Mat &img) {
   cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
   std::vector<cv::Point2f> corners;
   cv::goodFeaturesToTrack(img, corners, 500, 0.01, 3);
   for (auto &itr: corners) {
      cv::KeyPoint kpt;
      kpt.pt.x = itr.x;
      kpt.pt.y = itr.y;
      kps.push_back(kpt);
   }
   orb->compute(img, kps, des);
}

void Frame::set_pose(cv::Mat pose) {
   this->pose_w2c = pose;
   update_poses();
}

void Frame::update_poses() {
   pose_c2w  = pose_w2c.inv();
   cv::Mat R = pose_w2c.rowRange(0, 3).colRange(0, 3);
   cv::Mat t = pose_w2c.rowRange(0, 3).col(3);
   center    = -R.t() * t;
}

void Frame::add_observation(Point *point, size_t idx) {
   points[idx] = point;
}

bool Frame::compute_features(Features &features) {
    // Detect the corners
    feature_detection::FastDetector detector(_img_pyr[0].cols,
                                             _img_pyr[0].rows,
                                             Config::cell_size(),
                                             Config::n_pyr_levels());
    detector.detect((FramePtr)this, _img_pyr,
                    Config::min_harris_corner_score(),
                    features);

    if (features.size() < 100)
        return false;
    return true;
}

void Frame::add_observation(dr3::Feature *feature) {
    _fts.push_back(feature);
}

} // namespace dr3
