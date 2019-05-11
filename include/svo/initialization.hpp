#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include "global.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "features.hpp"
#include "viewer.hpp"

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
    Init() =default;
    ~Init() =default;

    Result add_first_frame(FramePtr frame_ref);
    Result add_second_frame(FramePtr frame_cur);

    double compute_inliers(const Matrix3d &R,
            const Vector3d &t);

protected:
    vector<cv::Point2f> _kps_ref;        // keypoints in the ref frame
    vector<cv::Point2f> _kps_cur;        // keypoints in the cur frame
    vector<Vector3d>    _pts_ref;        // bearing vectors in the ref frame
    vector<Vector3d>    _pts_cur;        // bearing vectors in the cur frame
    vector<Vector3d>    _xyz_in_cur;     // 3D points after homography
    vector<double>      _disparities;    // Disparities for each matching point
    vector<int>         _inliers;        // inlier indices
    SE3                 _T_cur_from_ref; // Transformation matrix (ref -> cur)
};

} // namespace init

} // namespace dr3

#endif // _INITIALIZATION_HPP_
