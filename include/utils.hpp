#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace utils {

const float PI = 3.141592653589793;
const int INF = INT_MAX;

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

cv::Mat compute_homography(std::vector<cv::KeyPoint> f1,
    std::vector<cv::KeyPoint> f2,
    std::vector<cv::DMatch> matches
) {
    int n_matches = matches.size();

    int n_rows = 2 * n_matches;
    int n_cols = 9;
    cv::Size A_shape(n_cols, n_rows);
    cv::Mat A = cv::Mat::zeros(A_shape, CV_32F);

    int index = 0;

    for (int i = 0; i < n_matches; ++i) {
        cv::DMatch match = matches[i];

        cv::Point2f p1 = f1[match.queryIdx].pt;
        cv::Point2f p2 = f2[match.trainIdx].pt;

        float ax = p1.x;
        float ay = p1.y;

        float bx = p2.x;
        float by = p2.y;

        float row1[] = {ax, ay, 1, 0, 0, 0, -bx*ax, -bx*ay, -bx};
        float row2[] = {0, 0, 0, ax, ay, 1, -by*ax, -by*ay, -by};

        cv::Mat r1(cv::Size(9, 1), CV_32F, &row1);
        cv::Mat r2(cv::Size(9, 1), CV_32F, &row2);

        r1.copyTo(A.row(index));
        r2.copyTo(A.row(index+1));

        index += 2;
    }

    // SVD
    cv::SVD svd(A, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
    cv::Mat Vt = svd.vt;
    cv::Mat H = cv::Mat::eye(cv::Size(3, 3), CV_32F);
    H = Vt.row(Vt.rows-1).reshape(1, 3);

    return H;
}

cv::Mat load_image(std::string file) {
    cv::Mat img = cv::imread(file, CV_LOAD_IMAGE_COLOR);
    
    if (!img.data) {
        std::cerr << "Couldn't open the image: " 
                  << file
                  << std::endl;
        exit(-1);
    }

    return img;
}

void view_image(std::string win_name, cv::Mat img) {
    cv::namedWindow(win_name, CV_WINDOW_FREERATIO);
    cv::imshow(win_name, img);
    cv::waitKey(0);
}

void save_image(std::string file_name, cv::Mat img) {
    cv::imwrite(file_name, img);
}

} // namespace utils

#endif
