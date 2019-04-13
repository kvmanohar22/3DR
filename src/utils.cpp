#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace utils {

cv::Mat translate(float Tx = 0, float Ty = 0) {
    cv::Mat T = cv::Mat::eye(cv::Size(3, 2), CV_32FC1);
    T.at<float>(0, 2) = Tx;
    T.at<float>(1, 2) = Ty;

    return T;
}

cv::Mat rotate(double theta=0, 
               double Rx = 0, 
               double Ry = 0) {
    cv::Mat R = cv::getRotationMatrix2D(cv::Point(cv::Size(Rx, Ry)), theta, 1.0);

    return R;
}

cv::Mat scale(double Sx = 1.0, double Sy = 1.0) {
    cv::Mat S = cv::Mat::eye(cv::Size(3, 2), CV_32FC1);
    S.at<float>(0, 0) = Sx;
    S.at<float>(1, 1) = Sy;

    return S;
}

cv::Mat compute_homography(std::vector<cv::KeyPoint> f1,
    std::vector<cv::KeyPoint> f2,
    std::vector<cv::DMatch> matches) {

    int n_matches = matches.size();

    int n_rows = 2 * n_matches;
    int n_cols = 9;
    cv::Size A_shape(n_cols, n_rows);
    cv::Mat A = cv::Mat::zeros(A_shape, CV_32F);

    int index = 0;

    for (int i = 0; i < n_matches; ++i) {
        cv::DMatch match = matches[i];

        cv::Point2f p1 = f1[match.queryIdx].pt;
        cv::Point2f p2 = f2[match.trainIdx].pt;

        float ax = p1.x;
        float ay = p1.y;

        float bx = p2.x;
        float by = p2.y;

        float row1[] = {ax, ay, 1, 0, 0, 0, -bx*ax, -bx*ay, -bx};
        float row2[] = {0, 0, 0, ax, ay, 1, -by*ax, -by*ay, -by};

        cv::Mat r1(cv::Size(9, 1), CV_32F, &row1);
        cv::Mat r2(cv::Size(9, 1), CV_32F, &row2);

        r1.copyTo(A.row(index));
        r2.copyTo(A.row(index+1));

        index += 2;
    }

    // SVD
    cv::SVD svd(A, cv::SVD::FULL_UV | cv::SVD::MODIFY_A);
    cv::Mat Vt = svd.vt;
    cv::Mat H = cv::Mat::eye(cv::Size(3, 3), CV_32F);
    H = Vt.row(Vt.rows-1).reshape(1, 3);

    // Transforms the points from `f1` to `f2`
    return H;
}

cv::Mat load_image(std::string file) {
    cv::Mat img = cv::imread(file, CV_LOAD_IMAGE_COLOR);
    
    if (!img.data) {
        std::cerr << "Couldn't open the image: " 
                  << file
                  << std::endl;
        exit(-1);
    }

    return img;
}

void view_image(std::string win_name, cv::Mat img) {
    cv::namedWindow(win_name, CV_WINDOW_NORMAL);
    cv::resizeWindow(win_name, 640, 640);
    cv::imshow(win_name, img);
    cv::waitKey(0);
}

void save_image(std::string file_name, cv::Mat img) {
    cv::imwrite(file_name, img);
}

template <typename T>
std::vector<T> arange(int start, int end) {
    const int N = end - start + 1; 
    std::vector<T> xs(N);
    for (auto &itr : xs) {
        itr = static_cast<T>(start++);
    }
    return xs;
}

void compute_spherical_warping(cv::Size2i out_shape, float f, cv::Mat &u, cv::Mat &v) {
    const int h = out_shape.height;
    const int w = out_shape.width;
    cv::Mat xf  = cv::Mat::zeros(cv::Size(w, h), CV_32F);
    cv::Mat yf  = cv::Mat::zeros(cv::Size(w, h), CV_32F);

    for (int idx = 0; idx < xf.rows; ++idx) {
        std::vector<float> vec = utils::arange<float>(0, w);
        float data[vec.size()];
        std::copy(vec.begin(), vec.end(), data);
        cv::Mat row(cv::Size(xf.cols, 1), CV_32F, &data);
        row.copyTo(xf.row(idx));
    }
    for (int idx = 0; idx < yf.cols; ++idx) {
        std::vector<float> vec = utils::arange<float>(0, h);
        float data[vec.size()];
        std::copy(vec.begin(), vec.end(), data);
        cv::Mat col(cv::Size(1, yf.rows), CV_32F, &data);
        col.copyTo(yf.col(idx));
    }

    #ifdef DEBUG
        std::cout << xf << std::endl;
        std::cout << yf << std::endl;
    #endif

    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            xf.at<float>(i, j) = (xf.at<float>(i, j) - 0.5 * w) / f;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            yf.at<float>(i, j) = (yf.at<float>(i, j) - 0.5 * h) / f;

    #ifdef DEBUG
        std::cout << xf << std::endl;
        std::cout << yf << std::endl;
    #endif

    cv::Mat xhat(cv::Mat::zeros(cv::Size(w, h), CV_32F));
    cv::Mat yhat(cv::Mat::zeros(cv::Size(w, h), CV_32F));
    cv::Mat zhat(cv::Mat::zeros(cv::Size(w, h), CV_32F));

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const float _xf = xf.at<float>(i, j); 
            const float _yf = yf.at<float>(i, j); 
            xhat.at<float>(i, j) = sin(_xf) * cos(_yf);
            yhat.at<float>(i, j) = sin(_yf);
            zhat.at<float>(i, j) = cos(_xf) * cos(_yf);
        }
    }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const float _zhat = zhat.at<float>(i, j); 
            xhat.at<float>(i, j) /= _zhat;
            yhat.at<float>(i, j) /= _zhat;
        }
    }

    u = 0.5 * w + xhat * f;
    v = 0.5 * h + yhat * f;
}

cv::Mat warp_local(cv::Mat img, cv::Mat &u, cv::Mat &v) {
    cv::Mat warped;
    cv::remap(img, warped, u, v, CV_INTER_LINEAR);

    return warped.clone();
}

cv::Mat warp_spherical(cv::Mat img, float f) {
    cv::Size size = img.size();
    cv::Mat u, v;
    compute_spherical_warping(size, f, u, v);
    return warp_local(img, u, v);
}

void compute_cylindrical_warping(cv::Size2i out_shape, float f, cv::Mat &u, cv::Mat &v) {
    const int h = out_shape.height;
    const int w = out_shape.width;
    cv::Mat theta  = cv::Mat::zeros(cv::Size(w, h), CV_32F);
    cv::Mat height  = cv::Mat::zeros(cv::Size(w, h), CV_32F);

    for (int idx = 0; idx < theta.rows; ++idx) {
        std::vector<float> vec = utils::arange<float>(0, w);
        float data[vec.size()];
        std::copy(vec.begin(), vec.end(), data);
        cv::Mat row(cv::Size(theta.cols, 1), CV_32F, &data);
        row.copyTo(theta.row(idx));
    }
    for (int idx = 0; idx < height.cols; ++idx) {
        std::vector<float> vec = utils::arange<float>(0, h);
        float data[vec.size()];
        std::copy(vec.begin(), vec.end(), data);
        cv::Mat col(cv::Size(1, height.rows), CV_32F, &data);
        col.copyTo(height.col(idx));
    }

    #ifdef DEBUG
        std::cout << theta << std::endl;
        std::cout << height << std::endl;
    #endif

    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            theta.at<float>(i, j) = (theta.at<float>(i, j) - 0.5 * w) / f;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            height.at<float>(i, j) = (height.at<float>(i, j) - 0.5 * h) / f;

    #ifdef DEBUG
        std::cout << theta << std::endl;
        std::cout << height << std::endl;
    #endif

    cv::Mat xhat(cv::Mat::zeros(cv::Size(w, h), CV_32F));
    cv::Mat yhat(cv::Mat::zeros(cv::Size(w, h), CV_32F));
    cv::Mat zhat(cv::Mat::zeros(cv::Size(w, h), CV_32F));

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const float _theta = theta.at<float>(i, j);
            const float _height = height.at<float>(i, j);
            xhat.at<float>(i, j) = sin(_theta);
            yhat.at<float>(i, j) = _height;
            zhat.at<float>(i, j) = cos(_theta);
        }
    }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const float _zhat = zhat.at<float>(i, j); 
            xhat.at<float>(i, j) /= _zhat;
            yhat.at<float>(i, j) /= _zhat;
        }
    }

    u = 0.5 * w + xhat * f;
    v = 0.5 * h + yhat * f;
}

cv::Mat warp_cylindrical(cv::Mat img, float f) {
    cv::Size size = img.size();
    cv::Mat u, v;
    compute_cylindrical_warping(size, f, u, v);
    return warp_local(img, u, v);
}

cv::Scalar getc() {
    float b = rand() % 255;
    float g = rand() % 255;
    float r = rand() % 255;

    return cv::Scalar(b, g, r);
}

float shi_tomasi_score(const cv::Mat &img, int u, int v) {
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;

    const int halfbox_size = 4;
    const int box_size = 2 * halfbox_size;
    const int box_area = box_size * box_size;
    const int x_min = u - halfbox_size;
    const int x_max = u + halfbox_size;
    const int y_min = v - halfbox_size;
    const int y_max = v + halfbox_size;

    // too close to the border
    if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
        return 0.0;

    const int stride = img.step.p[0];
    for(int y = y_min; y < y_max; ++y ) {
        const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx*dx;
            dYY += dy*dy;
            dXY += dx*dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
}

} // namespace utils

#endif
