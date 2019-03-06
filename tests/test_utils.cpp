#include "utils.hpp"

template <typename T>
void print_vec(std::vector<T> vec) {
    for (auto &v : vec)
        std::cout << v << " ";
    std::cout << std::endl;
}

int main() {
    using namespace std;

    // TEST LINSPACE
    cout << "Testing linspace...\n";
    std::vector<float> A;
    A = utils::linspace<float>(1, 5, 5);
    print_vec<float>(A);
    A = utils::linspace<float>(6, 23.4, 5);
    print_vec<float>(A);
    A = utils::linspace<float>(0, 2, 1);
    print_vec<float>(A);
    A = utils::linspace<float>(0, 2, 0);
    print_vec<float>(A);


    // CONCAT
    cout << "Testing concat...\n";
    std::vector<int> v1(10, 1);
    std::vector<int> v2(10, 2);
    std::vector<int> v3(10, 3);
    print_vec<int>(v1);
    print_vec<int>(v2);
    print_vec<int>(v3);

    v1.insert(v1.end(), v2.begin(), v2.end());
    v1.insert(v1.end(), v3.begin(), v3.end());
    print_vec<int>(v1);

    // TEST ARANGE
    cout << "Testing arange...\n";
    std::vector<float> B;
    B = utils::arange<float>(0, 10);
    print_vec<float>(B);
    B = utils::arange<float>(0, 1);
    print_vec<float>(B);

    // COPYING
    cout << "Testing copy functions...\n";
    #define DEBUG
    cv::Mat u, v;
    utils::compute_spherical_warping(cv::Size2i(10, 10), 1.0f, u, v);
    #undef DEBUG

    // TEST WARPING
    cout << "Testing spherical warping...\n";
    cv::Mat img = utils::load_image("../imgs/field/field1.jpg");
    utils::view_image("Field", img);
    cv::Mat warped = utils::warp_spherical(img, 700.0f);
    utils::view_image("Warped Field", warped);
}
