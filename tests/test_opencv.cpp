#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


int main() {
   using namespace std;

   cout << "Testing scalar multiplication..." << endl;
   float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
   cv::Mat mat(cv::Size(3, 3), CV_32F, &data);
   cout << "Initial matrix: \n" << mat << endl;
   mat *= 5;
   cout << "Final matrix: \n" << mat << endl;
   cout << mat << endl << endl;

   // Setting the values
   cout << "Setting values..." << endl;
   cv::Mat A(1, 5, CV_64FC2);
   cout << A.rows << " " << A.cols << endl;
   A.at<cv::Vec2d>(0, 0) = {1, 2};
   cout << A << endl << endl;

   std::vector<cv::Mat> Rt(4, cv::Mat(cv::Size(4, 3), CV_32F));
   cout << Rt[0].rowRange(0, 3).colRange(0, 3) << endl;
}
