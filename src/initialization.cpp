#include "svo/initialization.hpp"

#include <vikit/homography.h>

namespace dr3 {

namespace init {

Result Init::add_first_frame(const FramePtr frame_ref) {
    // Detect the corners
    feature_detection::FastDetector detector(frame_ref->_img_pyr[0].cols,
                                             frame_ref->_img_pyr[0].rows,
                                             Config::cell_size(),
                                             Config::n_pyr_levels());
    detector.detect(frame_ref, frame_ref->_img_pyr,
                    Config::min_harris_corner_score(),
                    frame_ref->_fts);

    if (frame_ref->_fts.size() < 100) {
        return Result::FAILED;
    }

    // Initialize the keypoints for the reference frame
    _kps_ref.clear(); _kps_ref.reserve(frame_ref->_fts.size());
    _pts_ref.clear(); _pts_ref.reserve(frame_ref->_fts.size());
    std::for_each(frame_ref->_fts.begin(), frame_ref->_fts.end(), [&](Feature *ftr) {
        _kps_ref.emplace_back(cv::Point2f(ftr->px[0], ftr->px[1]));
        _pts_ref.push_back(ftr->f);
        delete ftr;
    });

    _frame_ref = frame_ref;

    // Initialize the keypoints of future frame (cur) to these keypoints
    _kps_cur.insert(_kps_cur.begin(), _kps_ref.begin(), _kps_ref.end());
    return Result::SUCCESS;
}

Result Init::add_second_frame(const FramePtr frame_cur) {
    // KLT Tracker
    const int klt_win_size = 30;
    const int klt_max_iter = 1000;
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
            kps_ref_itr = _kps_ref.erase(kps_ref_itr);
            kps_cur_itr = _kps_cur.erase(kps_cur_itr);
            pts_ref_itr = _pts_ref.erase(pts_ref_itr);
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

    for (size_t i = 0; i < _pts_ref.size(); ++i) {
        uv_ref[i] = vk::project2d(_pts_ref[i]);
        uv_cur[i] = vk::project2d(_pts_cur[i]);
    }

    double ee = _frame_ref->_cam->error2();
    double rr = 30.0;

    // Draw the matches computed from optical flow
    Viewer2D::update(_frame_ref->_img_pyr[0],
                     frame_cur->_img_pyr[0],
                     _kps_ref, _kps_cur);

    // Compute homography on normalized image coordinates
    vk::Homography homography(uv_ref, uv_cur, ee, rr);
    homography.computeSE3fromMatches();
    vector<int> outliers;
//    double tot_error = vk::computeInliers(_pts_cur, _pts_ref,
//                       homography.T_c2_from_c1.rotation_matrix(),
//                       homography.T_c2_from_c1.translation(),
//                       rr, ee,
//                       _xyz_in_cur, _inliers, outliers);
    double tot_error = compute_inliers(homography.T_c2_from_c1.rotation_matrix(),
                                       homography.T_c2_from_c1.translation());
    _T_cur_from_ref = homography.T_c2_from_c1;

    std::cout << "#inliers: " << _inliers.size() << std::endl;
    std::cout << "reprojection error: " << tot_error << std::endl;
    std::cout << "reprojection threshold: " << rr << std::endl;
    std::cout << "Error multiplier: " << ee << std::endl;

    while (true) {
        Viewer2D::update(_frame_ref->_img_pyr[0],
                         frame_cur->_img_pyr[0],
                         _kps_ref, _kps_cur);
    }
}

double Init::compute_inliers(const Matrix3d &R, const Vector3d &t) {
    vector<int> outliers;
    const double reprojection_threshold = Config::reprojection_threshold();

    const size_t size = _pts_cur.size();
    double error = 0.0f;
    _inliers.clear(); _inliers.reserve(size);
    outliers.clear(); outliers.reserve(size);
    _xyz_in_cur.clear(); _xyz_in_cur.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        // Triangulate the point
        _xyz_in_cur.emplace_back(vk::triangulateFeatureNonLin(R, t,
                _pts_cur[i], _pts_ref[i]));

        // Reprojection error wrt current frame
        Vector3d xyz_in_cur = _xyz_in_cur.back();
        Vector2d uv_cur = vk::project2d(_pts_cur[i]);
        Vector2d uv_cur_rep = vk::project2d(xyz_in_cur);
        Vector2d _e1 = uv_cur - uv_cur_rep;
        double e1 = _e1.norm();

        // Reprojection error wrt reference frame
        Vector3d xyz_in_ref = R.transpose() * (xyz_in_cur - t);
        Vector2d uv_ref = vk::project2d(_pts_ref[i]);
        Vector2d uv_ref_rep = vk::project2d(xyz_in_ref);
        Vector2d _e2 = uv_ref - uv_ref_rep;
        double e2 = _e2.norm();

        bool is_inlier = false;
        if (e1 < reprojection_threshold && e2 < reprojection_threshold) {
            _inliers.emplace_back(i);
            is_inlier = true;
        } else {
            outliers.emplace_back(i);
        }

        // Compute the coordinates in the normal image space
        auto cam = (Pinhole*)_frame_ref->get_cam();
        uv_cur[0] = uv_cur[0] * cam->fx() + cam->cx();
        uv_cur[1] = uv_cur[1] * cam->fy() + cam->cy();
        uv_ref[0] = uv_ref[0] * cam->fx() + cam->cx();
        uv_ref[1] = uv_ref[1] * cam->fy() + cam->cy();
        uv_cur_rep[0] = uv_cur_rep[0] * cam->fx() + cam->cx();
        uv_cur_rep[1] = uv_cur_rep[1] * cam->fy() + cam->cy();
        uv_ref_rep[0] = uv_ref_rep[0] * cam->fx() + cam->cx();
        uv_ref_rep[1] = uv_ref_rep[1] * cam->fy() + cam->cy();

        cout << "idx: " << i << "      Inlier: " << is_inlier << endl;
        cout << "cur       : " << uv_cur.transpose() << endl;
        cout << "xyz_in_cur: " << uv_cur_rep.transpose() << endl;
        cout << "e1        : " << e1 << endl;
        cout << "ref       : " << uv_ref.transpose() << endl;
        cout << "xyz_in_ref: " << uv_ref_rep.transpose() << endl;
        cout << "e2        : " << e2 << endl;
        cout << "Total     : " << e1+e2 << endl;
        cout << "-------------------" << endl;

        error += (e1 + e2);
    }
    return error;
}

} // namespace init

} // namespace dr3
