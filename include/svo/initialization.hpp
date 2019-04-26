#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include "global.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "features.hpp"

#include <vector>

namespace dr3 {

namespace init {

enum class Result {
    FAILED,
    SUCCESS
};

class Init {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FramePtr _frame_ref;
    Init() {}
    ~Init() {}

    Result add_first_frame(FramePtr frame_ref);
    Result add_second_frame(FramePtr frame_cur);

protected:
    vector<cv::Point2f> _kps_ref; // keypoints in the ref frame
    vector<cv::Point2f> _kps_cur; // keypoints in the cur frame
    vector<Vector3d>    _pts_ref; // bearing vectors in the ref frame
    vector<Vector3d>    _pts_cur; // bearing vectors in the cur frame

    vector<double> _disparities;
    vector<int> _inliers;
};

} // namespace init

} // namespace dr3

#endif // _INITIALIZATION_HPP_
