#include "svo/initialization.hpp"

#include <vikit/homography.h>

namespace dr3 {

namespace init {

Result Init::add_first_frame(FramePtr frame_ref) {
    // Detect the corners
    feature_detection::FastDetector detector(frame_ref->_img_pyr[0].cols,
                                             frame_ref->_img_pyr[0].rows,
                                             Config::cell_size(),
                                             Config::n_pyr_levels());
    detector.detect(frame_ref, frame_ref->_img_pyr,
                    Config::min_harris_corner_score(),
                    frame_ref->_fts);

    // TODO: Check if the features are less than some threshold

    // Initialize the keypoints for the reference frame
    _kps_ref.clear(); _kps_ref.reserve(frame_ref->_fts.size());
    _pts_ref.clear(); _pts_ref.reserve(frame_ref->_fts.size());
    std::for_each(frame_ref->_fts.begin(), frame_ref->_fts.end(), [&](Feature *ftr) {
        _kps_ref.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
        _pts_ref.push_back(ftr->f);
        delete ftr;
    });

    _frame_ref = frame_ref;

    // Initialize the keypoints of future frame (cur) to these keypoints
    _kps_cur.insert(_kps_cur.begin(), _kps_ref.begin(), _kps_ref.end());
    return Result::SUCCESS;
}

Result Init::add_second_frame(FramePtr frame_cur) {
    // KLT Tracker
    const double klt_win_size = 30.0;
    const int klt_max_iter = 30;
    const double klt_eps = 1e-3;
    vector<uchar> status;
    vector<float> error;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                              klt_max_iter, klt_eps);
    cv::calcOpticalFlowPyrLK(_frame_ref->_img_pyr[0],
                             frame_cur->_img_pyr[0],
                             _kps_ref, _kps_cur,
                             status, error,
                             cv::Size2i(klt_win_size, klt_win_size),
                             4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

    auto kps_ref_itr = _kps_ref.begin();
    auto kps_cur_itr = _kps_cur.begin();
    auto pts_ref_itr = _pts_ref.begin();
    _disparities.clear(); _disparities.reserve(_kps_cur.size());
    _pts_cur.clear(); _pts_cur.reserve(_kps_cur.size());
    for (size_t i = 0; kps_ref_itr != _kps_ref.end(); ++i) {
        if (!status[i]) {
            _kps_ref.erase(kps_ref_itr);
            _kps_cur.erase(kps_cur_itr);
            _pts_ref.erase(pts_ref_itr);
            continue;
        }
        _disparities.push_back(Vector2d(kps_ref_itr->x - kps_cur_itr->x,
                               kps_ref_itr->y - kps_cur_itr->y).norm());
        _pts_cur.push_back(frame_cur->_cam->cam2world(kps_cur_itr->x, kps_cur_itr->y));
        ++kps_ref_itr;
        ++kps_cur_itr;
        ++pts_ref_itr;
    }

    // Compute homography
    vector<Vector2d> uv_ref(_pts_ref.size());
    vector<Vector2d> uv_cur(_pts_cur.size());
    std::cout << "ref: " << uv_ref.size() << std::endl;
    std::cout << "cur: " << uv_cur.size() << std::endl;
    std::cout << "e2: " << _frame_ref->_cam->error2() << std::endl;

    int jj=0;
    for (size_t i = 0; i < _pts_ref.size(); ++i) {
        uv_ref[i] = vk::project2d(_pts_ref[i]);
        uv_cur[i] = vk::project2d(_pts_cur[i]);
        ++jj;
    }

    std::cout << "ref: " << uv_ref.size() << std::endl;
    std::cout << "cur: " << uv_cur.size() << std::endl;
    std::cout << "jj: " << jj << std::endl;

    vk::Homography homography(uv_ref, uv_cur,
                              718.0f,
                              5.0f);

    // homography.computeSE3fromMatches();
    // vector<int> outliers;
    // vector<Vector3d> xyz_in_cur;
    // vk::computeInliers(_pts_cur, _pts_ref,
    //                    homography.T_c2_from_c1.rotation_matrix(),
    //                    homography.T_c2_from_c1.translation(),
    //                    5.0f, _frame_ref->_cam->error2(),
    //                    xyz_in_cur, _inliers, outliers);

}

} // namespace init

} // namespace dr3