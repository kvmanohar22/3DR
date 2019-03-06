#include "panorama.hpp"

#include <string>
#include <sstream>

int main() {
    using namespace std;

    vector<string> image_sets = {"field", "yosemite", "rainer"};
    stringstream ss;
    ss << "../imgs/" << image_sets[1];
    std::string dir = ss.str();
    const char *c_str = dir.c_str();

    // NORMAL PANORAMA STITCHING
    reconstruct::Panorama pan_gen(595.0f, -0.15f, 0.00f, 50);
    std::cout << "Focal length: " << pan_gen.get_focal_length() << std::endl;
    std::cout << "K1: " << pan_gen.get_k1() << std::endl;
    std::cout << "K2: " << pan_gen.get_k2() << std::endl;
    std::cout << "Feathering width: " << pan_gen.get_feathering_width() << std::endl;

    int flag = pan_gen.process(c_str);
    if (flag == 0) {
        cv::Mat pan = pan_gen.get_final_panorama();
        utils::view_image("panorama", pan);
        utils::save_image("../imgs/results/panorama/"+image_sets[1]+".jpg", pan);
    }
}
