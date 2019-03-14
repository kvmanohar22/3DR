#include "two.hpp"

int main() {
   using namespace std;
   using namespace dr3;

   // images
   string img_l = "../imgs/slam/img_l.png";
   string img_r = "../imgs/slam/img_r.png";

   #define DEBUG
   TwoView twoview(img_l, img_r);

   cv::Mat F = twoview.estimate_F();
   cout << "F: \n" << F << endl;

   cv::SVD svd(F, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
   cout << svd.w << endl;
}