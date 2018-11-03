#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

int main()
{
    cv::Mat img = cv::imread("../imgs/triangle1.jpg");

    if (!img.data)
    {
        std::cout << "Couldn't load the image" << std::endl;
    }

    cv::Mat R = utils::rotate(20, 60, 60);

    cv::Point p1(cv::Size(40, 40));
    cv::Point p2(cv::Size(80, 40));
    cv::Point p3(cv::Size(80, 80));
    cv::Point p4(cv::Size(40, 80));
    cv::Point p5(cv::Size(60, 60));
    cv::Point p6(cv::Size(80, 60));

    cv::line(img, p1, p2, cv::Scalar(255, 255, 255));
    cv::line(img, p2, p3, cv::Scalar(255, 255, 255));
    cv::line(img, p3, p4, cv::Scalar(255, 255, 255));
    cv::line(img, p4, p1, cv::Scalar(255, 255, 255));
    cv::line(img, p5, p6, cv::Scalar(255, 255, 255));

    cv::namedWindow("image", CV_WINDOW_FREERATIO);
    cv::imshow("image", img);

    cv::Mat out;
    cv::warpAffine(img, out, R, cv::Size(200, 200));
    cv::namedWindow("warped", CV_WINDOW_FREERATIO);
    cv::imshow("warped", out);

    cv::waitKey(0);
    return 0;
}