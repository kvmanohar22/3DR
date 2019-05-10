#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "features.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "camera.hpp"
#include "viewer.hpp"

using namespace std;
using namespace dr3;
using namespace dr3::feature_detection;

int main() {
    cv::Mat img = cv::imread("../imgs/sample.jpg", 0);

    if (!img.data) {
        std::cout << "Couldn't load the image" << std::endl;
    }

    {

    // camera
    AbstractCamera *cam = new Pinhole(1240, 376,
                                      718.856, 718.856,
                                      607.1928, 185.2157);
    cv::Mat K = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
    FramePtr frame(new Frame(0, img, cam));

    Features new_features;
    ImgPyramid pyr;
    utils::create_img_pyramid(img, Config::n_pyr_levels(), pyr);
    FastDetector detector(img.cols,
                          img.rows,
                          Config::cell_size(),
                          Config::n_pyr_levels());
    cout << "n_pyr_levels: " << Config::n_pyr_levels() << endl;
    cout << "min harris score: " << Config::min_harris_corner_score() << endl;
    cout << "Number of detected features: " << new_features.size() << endl;
    detector.detect(frame, pyr, Config::min_harris_corner_score(), new_features);

    cv::cvtColor(img, img, CV_GRAY2BGR);
    std::for_each(new_features.begin(), new_features.end(), [&](Feature *ftr) {
        cv::Scalar color = utils::getc();
        dr3::Viewer2D::draw_point(img,
                                  cv::Point2f(ftr->px[0], ftr->px[1]),
                                  color);
    });
    
    }

    cv::imshow("fast corners", img);
    cv::waitKey(0);

    return 0;
}