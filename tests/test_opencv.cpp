#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


void fill_random(cv::Mat &mat) {
   for (int i = 0; i < mat.rows; ++i)
      for (int j = 0; j < mat.cols; ++j)
         mat.at<float>(i, j) = rand() % 45;
}

int main() {
   using namespace std;

   // cout << "Testing scalar multiplication..." << endl;
   // float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
   // cv::Mat mat(cv::Size(3, 3), CV_32F, &data);
   // cout << "Initial matrix: \n" << mat << endl;
   // mat *= 5;
   // cout << "Final matrix: \n" << mat << endl;
   // cout << mat << endl << endl;

   // // Setting the values
   // cout << "Setting values..." << endl;
   // cv::Mat A(1, 5, CV_64FC2);
   // cout << A.rows << " " << A.cols << endl;
   // A.at<cv::Vec2d>(0, 0) = {1, 2};
   // cout << A << endl << endl;

   // std::vector<cv::Mat> Rt(4, cv::Mat(cv::Size(4, 3), CV_32F));
   // cout << Rt[0].rowRange(0, 3).colRange(0, 3) << endl;

   // // Rows and cols
   // cv::Mat pts4d1(4, 15, CV_32F);
   // cv::Mat pts4d2(cv::Size(4, 15), CV_32F);
   // // These above two have different behaviours
   // std::cout << pts4d.rows << " "
   //           << pts4d.cols << " "
   //           << std::endl;
   // fill_random(pts4d);
   // std::cout << pts4d << std::endl;

   // int j = 5;
   // float x, y, z, w;
   // x = pts4d.at<float>(0, j);
   // y = pts4d.at<float>(1, j);
   // z = pts4d.at<float>(2, j);
   // w = pts4d.at<float>(3, j);

   // std::cout << x << " "
   //           << y << " "
   //           << z << " "
   //           << w << " "
   //           << std::endl;

   // x = pts4d.at<float>(j, 0);
   // y = pts4d.at<float>(j, 1);
   // z = pts4d.at<float>(j, 2);
   // w = pts4d.at<float>(j, 3);

   // std::cout << x << " "
   //           << y << " "
   //           << z << " "
   //           << w << " "
   //           << std::endl;

   // float inf = 1.0 / 0.0;
   // cerr << inf << endl;
   // float nan1 = 0.0 * inf;
   // float nan2 = 0.0 * inf;
   
   // if (nan1 == nan2)
   //    cerr << nan << endl;
   // cerr << nan1 << endl;

   // // copying is weird in opencv
   // std::vector<cv::Mat> Rt(4, cv::Mat(cv::Size(4, 3), CV_32F));
   cv::Mat xyz(cv::Size(1, 3), CV_32F);
   xyz.at<float>(0) = 12;
   xyz.at<float>(1) = 132;
   xyz.at<float>(2) = 2;
   std::cout << xyz << std::endl;
}
