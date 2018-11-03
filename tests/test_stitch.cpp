#include "stitch.hpp"
using namespace reconstruct;

int main() {
    reconstruct::Stitch stitcher(678.0f, -0.21f, 0.26f);

    std::string left("../imgs/yosemite1.jpg");
    std::string right("../imgs/yosemite2.jpg");

    stitcher.process(left, right);
}