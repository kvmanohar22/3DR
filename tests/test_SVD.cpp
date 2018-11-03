#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

int main()
{

    cv::Mat mat1 = cv::Mat::zeros(cv::Size(5, 4), CV_64F);

    mat1.at<double>(0, 0) = 1;
    mat1.at<double>(0, 4) = 2;
    mat1.at<double>(1, 2) = 3;
    mat1.at<double>(3, 1) = 2;

    std::cout << mat1 << std::endl;

    cv::SVD svd(mat1, cv::SVD::FULL_UV);
    std::cout << svd.w << std::endl;
    std::cout << svd.u << std::endl;
    std::cout << svd.vt << std::endl;

    std::cout << svd.vt.row(svd.vt.rows-1) << std::endl;
}
