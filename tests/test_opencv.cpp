#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


int main() {
   using namespace std;

   cout << "Testing scalar multiplication..." << endl;
   float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
   cv::Mat mat(cv::Size(3, 3), CV_32F, &data);
   cout << mat << endl;
   mat *= 5;
   cout << mat << endl;
}
