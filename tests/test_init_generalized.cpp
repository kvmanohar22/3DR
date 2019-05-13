//
// Created by kv on 12/5/19.
//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "features.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "viewer.hpp"
#include "camera.hpp"
#include "svo/initialization.hpp"

using namespace std;
using namespace dr3;
using namespace dr3::feature_detection;

void draw_features(Features &features, cv::Mat &img) {
    std::for_each(features.begin(), features.end(), [&](Feature *ftr) {
        cv::Scalar color = utils::getc();
        dr3::Viewer2D::draw_point(img,
                                  cv::Point2f(ftr->px[0], ftr->px[1]),
                                  color);
    });
}

int main() {
    cv::Mat img1 = cv::imread("../imgs/kitti0.png", 0);
    cv::Mat img2 = cv::imread("../imgs/kitti5.png", 0);

    if (!img1.data || !img2.data) {
        std::cout << "Couldn't load the image" << std::endl;
        return -2;
    }

    // camera
    AbstractCamera *cam = new Pinhole(1240, 376,
                                      718.856, 718.856,
                                      607.1928, 185.2157);

    FramePtr frame_ref(new Frame(0, img1, cam));
    FramePtr frame_cur(new Frame(7, img2, cam));

    Features new_features1, new_features2;
    FastDetector detector(img1.cols,
                          img1.rows,
                          Config::cell_size(),
                          Config::n_pyr_levels());

    // Detect features
    detector.detect(frame_ref,
                    frame_ref->_img_pyr,
                    Config::min_harris_corner_score(),
                    new_features1);
    detector.detect(frame_cur,
                    frame_cur->_img_pyr,
                    Config::min_harris_corner_score(),
                    new_features2);

    // draw features
    cv::cvtColor(img1, img1, CV_GRAY2BGR);
    cv::cvtColor(img2, img2, CV_GRAY2BGR);
    draw_features(new_features1, img1);
    draw_features(new_features2, img2);

    cout << "Ref frame feature count: " << new_features1.size() << endl;
    cout << "Cur frame feature count: " << new_features2.size() << endl;

    // Initial map generator
    init::Init initializer;
    initializer.add_first_frame(frame_ref);
    initializer.add_second_frame_generalized(frame_cur);

    return 0;
}
