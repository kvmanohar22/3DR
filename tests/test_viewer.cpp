#include "viewer.hpp"
#include "utils.hpp"

using namespace std;
using namespace dr3;

int main() {

   Viewer2D v2d;
   cv::Mat img = utils::load_image("../imgs/slam/img_l.png");

   // draw point
   cv::Point2f pt2d;
   pt2d.x = 323;
   pt2d.y = 133;
   v2d.draw_point(img, pt2d);
   utils::view_image("img", img);

   cv::Point3f pt3d;
   pt3d.x = 1;
   pt3d.y = 1;
   pt3d.z = 500;
   v2d.draw_line(img, pt3d);
   utils::view_image("img", img);
}
