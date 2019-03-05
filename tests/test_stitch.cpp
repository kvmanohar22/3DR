#include "stitch.hpp"
using namespace reconstruct;

int main() {
    reconstruct::Stitch stitcher(678.0f, -0.21f, 0.26f);
    std::string left, right;
    int flag;
    std::cout << "Focal length: " << stitcher.get_focal_length() << std::endl;
    std::cout << "K1: " << stitcher.get_k1() << std::endl;
    std::cout << "K2: " << stitcher.get_k2() << std::endl;

    // Image 1 and 2
    left = "../imgs/yosemite/yosemite1.jpg";
    right = "../imgs/yosemite/yosemite2.jpg";
    flag = stitcher.process(left, right);
    if (flag == 0) {
        cv::Mat img = stitcher.get_final_img();
        utils::view_image("stitch", img);
        utils::save_image("../imgs/results/stitch/y12.jpg", img);
    } else {
        std::cerr << "Stitching couldn't be performed";
    }

    // Image 2 and 3
    left = "../imgs/yosemite/yosemite2.jpg";
    right = "../imgs/yosemite/yosemite3.jpg";
    flag = stitcher.process(left, right);
    if (flag == 0) {
        cv::Mat img = stitcher.get_final_img();
        utils::view_image("stitch", img);
        utils::save_image("../imgs/results/stitch/y23.jpg", img);
    } else {
        std::cerr << "Stitching couldn't be performed";
    }

    // Image 3 and 4
    left = "../imgs/yosemite/yosemite3.jpg";
    right = "../imgs/yosemite/yosemite4.jpg";
    flag = stitcher.process(left, right);
    if (flag == 0) {
        cv::Mat img = stitcher.get_final_img();
        utils::view_image("stitch", img);
        utils::save_image("../imgs/results/stitch/y34.jpg", img);
    } else {
        std::cerr << "Stitching couldn't be performed";
    }

    // Image 3 and 4
    left = "../imgs/field/field1.jpg";
    right = "../imgs/field/field2.jpg";
    flag = stitcher.process(left, right);
    if (flag == 0) {
        cv::Mat img = stitcher.get_final_img();
        utils::view_image("stitch", img);
        utils::save_image("../imgs/results/stitch/f12.jpg", img);
    } else {
        std::cerr << "Stitching couldn't be performed";
    }
}
