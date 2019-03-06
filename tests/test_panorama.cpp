#include "panorama.hpp"

#include <string>
#include <sstream>

int main() {
    using namespace std;

    vector<string> image_sets = {"field", "yosemite", "rainer"};

    // Set these values
    const int idx = 0;
    const float f = 2600.0f;

    stringstream ss;
    ss << "../imgs/" << image_sets[idx];
    std::string dir = ss.str();
    const char *c_str = dir.c_str();
    int flag;
    stringstream result_name;
    result_name << "../imgs/results/panorama/" << image_sets[idx] << "_focal_length_" << f;

    // NORMAL PANORAMA STITCHING
    cout << "Normal Panorama stitching..." << endl;
    reconstruct::Panorama pan_gen(595.0f, -0.15f, 0.00f, 50);
    std::cout << "Focal length: " << pan_gen.get_focal_length() << std::endl;
    std::cout << "K1: " << pan_gen.get_k1() << std::endl;
    std::cout << "K2: " << pan_gen.get_k2() << std::endl;
    std::cout << "Feathering width: " << pan_gen.get_feathering_width() << std::endl;

    flag = pan_gen.process(c_str);
    if (flag == 0) {
        cv::Mat pan = pan_gen.get_final_panorama();
        utils::view_image("Normal panorama", pan);
        result_name << "_normal.jpg";
        string name = result_name.str();
        utils::save_image(name, pan);
    }

    // SPHERICAL PANORAMA STITCHING
    cout << "Spherical Panorama stitching..." << endl;
    reconstruct::Panorama pan_gen2(f, -0.15f, 0.00f, 50, reconstruct::PanType::Translate);
    std::cout << "Focal length: " << pan_gen2.get_focal_length() << std::endl;
    std::cout << "K1: " << pan_gen2.get_k1() << std::endl;
    std::cout << "K2: " << pan_gen2.get_k2() << std::endl;
    std::cout << "Feathering width: " << pan_gen2.get_feathering_width() << std::endl;
    flag = pan_gen2.process(c_str);
    if (flag == 0) {
        cv::Mat pan = pan_gen2.get_final_panorama();
        utils::view_image("Spherical panorama", pan);
        result_name << "_spherical.jpg";
        string name = result_name.str();
        utils::save_image(name, pan);
    }
}
