#include "utils.hpp"

template <typename T>
void print_vec(std::vector<T> vec) {
    for (auto &v : vec)
        std::cout << v << " ";
    std::cout << std::endl;
}

int main() {
    // TEST LINSPACE
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
    std::vector<int> v1(10, 1);
    std::vector<int> v2(10, 2);
    std::vector<int> v3(10, 3);
    print_vec<int>(v1);
    print_vec<int>(v2);
    print_vec<int>(v3);

    v1.insert(v1.end(), v2.begin(), v2.end());
    v1.insert(v1.end(), v3.begin(), v3.end());
    print_vec<int>(v1);
}