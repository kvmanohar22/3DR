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

   cv::Mat imgl, imgr;
   cv::Mat F = twoview._estimate_F();
   imgl = twoview.draw_poles_and_lines(100);
   utils::save_image("../imgs/slam/epipoles_left.png", imgl);

   imgr = twoview.draw_poles_and_lines(100, false);
   utils::save_image("../imgs/slam/epipoles_right.png", imgr);
}
