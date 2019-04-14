#include "svo/initialization.hpp"

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
    _kps_ref.clear();
    _kps_ref.reserve(frame_ref->_fts.size());
    std::for_each(frame_ref->_fts.begin(), frame_ref->_fts.end(), [&](Feature *ftr) {
        _kps_ref.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
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
}

} // namespace init

} // namespace dr3
