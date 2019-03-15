#include "two.hpp"
#include "utils.hpp"

int main() {
   using namespace std;
   using namespace dr3;

   // images
   string img_l = "../imgs/slam/img_l.png";
   string img_r = "../imgs/slam/img_r.png";

   #define DEBUG
   TwoView twoview(img_l, img_r);

   cv::Mat F = twoview.estimate_F();
   cv::Mat img = twoview.draw_poles_and_lines(100);
   utils::save_image("../imgs/slam/epipoles.png", img);
}