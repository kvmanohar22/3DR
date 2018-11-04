#include "stitch.hpp"
using namespace reconstruct;

int main() {
    reconstruct::Stitch stitcher(678.0f, -0.21f, 0.26f);

    std::string left("../imgs/yosemite1.jpg");
    std::string right("../imgs/yosemite2.jpg");

    std::cout << "Focal length: " << stitcher.get_focal_length() << std::endl;
    std::cout << "K1: " << stitcher.get_k1() << std::endl;
    std::cout << "K2: " << stitcher.get_k2() << std::endl;

    int flag = stitcher.process(left, right);
    
    if (flag == 0) {
        cv::Mat img = stitcher.get_final_img();
        utils::view_image("stitch", img);
    } else {
        std::cerr << "Stitching couldn't be performed";
    }
}