#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"


cv::Mat fuck_his_mother() {
    cv::Mat ones = cv::Mat::eye(cv::Size(3, 3), CV_32F);
    ones.at<float>(1, 1) = 122445;

    std::cout << ones << std::endl;
    return ones;
}


int main()
{

    // cv::Mat mat1 = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
    // cv::Mat mat2 = cv::Mat::ones(cv::Size(3, 3), CV_32FC1);

    // // std::cout << mat1 << std::endl;
    // // std::cout << mat2 << std::endl;

    // mat1.row(0).copyTo(mat2.row(2));

    // // std::cout << mat1 << std::endl;
    // // std::cout << mat2 << std::endl;

    // float data[] = {1, 2, 3};
    // cv::Mat new_mat(cv::Size(3, 1), CV_32F, &data);
    // // std::cout << new_mat << std::endl;

    // // std::cout << mat1 << std::endl;
    // new_mat.copyTo(mat1.row(1));
    // // std::cout << mat1 << std::endl;

    // const int k = 1;
    // cv::Mat img;
    // try
    // {
    //     img = cv::imread("img.png");
    // }
    // catch(const std::exception& e)
    // {
    //     std::cerr << e.what() << '\n';
    // }
    
    // // std::cout << "What?" << std::endl;


    // cv::Mat M = cv::Mat::eye(cv::Size(3, 3), CV_32F);
    // float query[] = {1, 2, 3};
    // M.at<float>(0, 1) = 2.0f;
    // M.at<float>(1, 1) = 3.0f;
    // M.at<float>(2, 1) = 3.0f;
    // cv::Mat query_mat(cv::Size(1, 3), CV_32F, &query);

    // // std::cout << M.type() << std::endl;
    // // std::cout << query_mat.type() << std::endl;

    // // std::cout << M << std::endl;
    // // std::cout << query_mat << std::endl;
    // cv::Mat final = M * query_mat;

    // // std::cout << M << std::endl;
    // // std::cout << final << std::endl;
    
    // float *ptr = final.ptr<float>(0);
    // // std::cout << ptr[0]
    //           << ptr[1]
    //           << ptr[2]
    //           << std::endl;

    // float w = 1;
    // float h = 3;
    // float points[][9] = {0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 34, 1, w, 9, 1, 23, h, 1,
    //                      0, 44, 11, w, 91, 11, 213, h, 11};
    // cv::Mat points_mat(cv::Size(9, 9), CV_32F, &points);

    // std::cout << points_mat << std::endl;
    // std::cout << points_mat.row(points_mat.rows-1).reshape(1, 3) << std::endl;

    // std::vector<int> v1 = {1, 2, 3, 5};
    // std::vector<int> v2 = {1, 233, 3333};
    // std::vector<int> v3 = {3, 32, 3};

    // for (int i = 0; i < v1.size(); ++i)
    //     std::cout << v1[i] << " ";
    // std::cout << std::endl;
    // v1 = v2;
    // for (int i = 0; i < v1.size(); ++i)
    //     std::cout << v1[i] << " ";
    // std::cout << std::endl;
    // v1 = v3;
    // for (int i = 0; i < v1.size(); ++i)
    //     std::cout << v1[i] << " ";
    // std::cout << std::endl;

    // cv::Mat ii;
    // float fin_Data[][3] = {1,2,3,4,5,6,7,8,9};
    // ii = cv::Mat(cv::Size(3, 3), CV_32F, &data);

    // std::cout << ">>>>>>>>>>>>>>>>\n";
    // std::cout << ii << std::endl;

    // std::vector<int> vec(2, 1);
    // for (auto &x : vec)
    //     std::cout << x << " ";
    // std::cout << std::endl;


    // cv::Mat what = fuck_his_mother();
    // std::cout << what << std::endl;
    cv::Mat ii = cv::Mat::zeros(cv::Size(3, 3), CV_32FC4);
    std::cout << ii.at<cv::Vec4f>(1, 1)[0] << std::endl;
    std::cout << ii.at<cv::Vec4f>(1, 1)[1] << std::endl;
    std::cout << ii.at<cv::Vec4f>(1, 1)[2] << std::endl;
    std::cout << ii.at<cv::Vec4f>(1, 1)[3] << std::endl;

    return 0;
}