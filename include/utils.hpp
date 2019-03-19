#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cstdlib>

namespace utils {

const float PI = 3.141592653589793;
const int INF = INT_MAX;

// image transformations
cv::Mat translate(float Tx = 0, float Ty = 0);
cv::Mat rotate(double theta=0, double Rx = 0, double Ry = 0);
cv::Mat scale(double Sx = 1.0, double Sy = 1.0);
cv::Mat compute_homography(std::vector<cv::KeyPoint> f1,
    std::vector<cv::KeyPoint> f2,
    std::vector<cv::DMatch> matches);

// Image I/O
cv::Mat load_image(std::string file);
void view_image(std::string win_name, cv::Mat img);
void save_image(std::string file_name, cv::Mat img);

// numpy-type functions
template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    T val = a;
    for (auto &itr : xs) {
        itr = val;
        val += h;
    }
    return xs;
}

template <typename T>
std::vector<T> arange(int start, int end);

// Spherical warping functions
void compute_spherical_warping(cv::Size2i out_shape, float f, cv::Mat &u, cv::Mat &v); 
cv::Mat warp_local(cv::Mat img, cv::Mat &u, cv::Mat &v);
cv::Mat warp_spherical(cv::Mat img, float f);

// Cylindrical warping functions
void compute_cylindrical_warping(cv::Size2i out_shape, float f, cv::Mat &u, cv::Mat &v); 
cv::Mat warp_cylindrical(cv::Mat img, float f);

// Drawing
cv::Scalar getc();

// misc
void remove_nans(cv::Mat &mat);

} // namespace utils

#endif
