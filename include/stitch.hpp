#ifndef _STITCH_HPP_
#define _STITCH_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include "utils.hpp"

namespace reconstruct {

class Stitch {

private:
    // camera parameters
    float _focal_length;
    float _k1;
    float _k2;

    // images
    cv::Mat _left_img;
    cv::Mat _right_img;

    // optimization parameters
    int ransac_iters;
    float ransac_thresh;
    int min_matches;

public:
    Stitch() : _focal_length(utils::INF), _k1(utils::INF), _k2(utils::INF) {
        _focal_length = utils::INF;
        _k1 = utils::INF;
        _k2 = utils::INF;
        ransac_iters = 500;
        ransac_thresh = 5.0f;
        min_matches = 4;
    }

    Stitch(float _focal_length, float _k1, float _k2) : ransac_iters(500), 
        ransac_thresh(5.0) {
        this->_focal_length = _focal_length;
        this->_k1 = _k1;
        this->_k2 = _k2;
        this->ransac_iters = 500;
        this->ransac_thresh = 5.0f;
        this->min_matches = 4;
    }

    ~Stitch() {}

    static inline bool comparator(cv::DMatch m1, cv::DMatch m2) {
        return m1.distance > m2.distance;
    }

    inline int is_valid() {
        return _focal_length != utils::INF && 
        _k1 != utils::INF && 
        _k2 != utils::INF;
    }

    inline void load_left_image(std::string file) {
        _left_img = utils::load_image(file);
    }

    inline void load_right_image(std::string file) {
        _right_img = utils::load_image(file);
    }

    int process(std::string left_image_file,
        std::string right_image_file);

    cv::Mat align_pair(std::vector<cv::KeyPoint> left_keypoints,
        std::vector<cv::KeyPoint> right_keypoints,
        std::vector<cv::DMatch> matches);

    std::vector<int> get_inliers(std::vector<cv::KeyPoint> f1,
        std::vector<cv::KeyPoint> f2,
        std::vector<cv::DMatch> matches,
        cv::Mat M);

    cv::Mat least_squares_fit(std::vector<cv::KeyPoint> f1,
        std::vector<cv::KeyPoint> f2,
        std::vector<cv::DMatch> matches,
        std::vector<int> best_inliers);

}; // class Stitch

} // namespace reconstruct

#endif