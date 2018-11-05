#include "panorama.hpp"

int main()
{
    reconstruct::Panorama pan_gen(678.0f, -0.21f, 0.26f, 50);

    std::string dir("../imgs/yosemite");

    std::cout << "Focal length: " << pan_gen.get_focal_length() << std::endl;
    std::cout << "K1: " << pan_gen.get_k1() << std::endl;
    std::cout << "K2: " << pan_gen.get_k2() << std::endl;
    std::cout << "Feathering width: " << pan_gen.get_feathering_width() << std::endl;

    const char *c_str = dir.c_str();
    int flag = pan_gen.process(c_str);

    if (flag == 0) {
        cv::Mat pan = pan_gen.get_final_panorama();
        utils::view_image("panorama", pan);
        utils::save_image("panorama.jpg", pan);
    }
}