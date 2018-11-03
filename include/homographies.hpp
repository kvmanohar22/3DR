#ifndef _HOMOGRAPHIES_HPP_
#define _HOMOGRAPHIES_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace utils {

cv::Mat translate(float Tx = 0, float Ty = 0) {
    cv::Mat T = cv::Mat::eye(cv::Size(3, 2), CV_32FC1);
    T.at<float>(0, 2) = Tx;
    T.at<float>(1, 2) = Ty;

    return T;
}

cv::Mat rotate(double theta=0, double Rx = 0, double Ry = 0) {
    cv::Mat R = cv::getRotationMatrix2D(cv::Point(cv::Size(Rx, Ry)), theta, 1.0);

    return R;
}

cv::Mat scale(double Sx = 1.0, double Sy = 1.0) {
    cv::Mat S = cv::Mat::eye(cv::Size(3, 2), CV_32FC1);
    S.at<float>(0, 0) = Sx;
    S.at<float>(1, 1) = Sy;

    return S;
}

}

#endif
