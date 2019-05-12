#include "stitch.hpp"

using namespace reconstruct;
using namespace std;

int main(int argc, char **argv) {
    // Parameter sets
    float f, k1, k2;
    string left, right, savefile;

    // Yosemite
    // f = 678.0; k1 = -0.21; k2 = 0.26;
    // left = "../imgs/yosemite/yosemite1.jpg";
    // right = "../imgs/yosemite/yosemite2.jpg";
    // savefile = "../imgs/results/stitch/yosemite12.jpg";

    // Drone
    f = 315.5; k1 = 0.0; k2 = 0.0;
    left  = "../imgs/drone/00.png";
    right = "../imgs/drone/04.png";
    savefile = "../imgs/results/stitch/drone04.jpg";

//    KITTI
//    f = 718.856; k1 = 0.0; k2 = 0.0;
//    left  = "../imgs/KITTI/000000.png";
//    right = "../imgs/KITTI/000007.png";
//    savefile = "../imgs/results/stitch/KITTI07.jpg";

    reconstruct::Stitch stitcher(f, k1, k2);
    std::cout << "Focal length: " << stitcher.get_focal_length() << std::endl;
    std::cout << "K1: " << stitcher.get_k1() << std::endl;
    std::cout << "K2: " << stitcher.get_k2() << std::endl;

    int flag;
    flag = stitcher.process(left, right);
    if (flag == 0) {
        cv::Mat img = stitcher.get_final_img();
        utils::view_image("stitch", img);
        utils::save_image(savefile, img);
    } else {
        std::cerr << "Stitching couldn't be performed";
    }
}
