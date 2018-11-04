#include "panorama.hpp"

namespace reconstruct {

int Panorama::process(char *dir_name) {

    // Load the images
    this->load_images(dir_name);
    std::vector<ImageInfo> _info;

    cv::Mat H_init = cv::Mat::eye(cv::Size(3, 3), CV_32F);
    for (int i = 0; i < this->_images.size()-1; ++i) {
        Stitch stitcher(_focal_length, _k1, _k2);

        ImageInfo img_info;
        img_info.set_img(_images[i]);
        img_info.set_H(H_init);

        int flag = stitcher.process(_images[i], _images[i+1]);
  
        if (flag == 0) {
            cv::Mat H_curr = stitcher.get_H();
            H_init = H_init * H_curr;
        } else {
            std::cerr << "Couldn't process images: "
                      << i << i+1 << std::endl;
        }
        _info.push_back(img_info);
    }
    ImageInfo img_info;
    img_info.set_img(_images[_images.size()-1]);
    img_info.set_H(H_init);
    _info.push_back(img_info);

    // Compute the canvas size
    this->set_canvas_size(_info);

    // Paste images
    this->paste_images(_info);
}

void Panorama::set_canvas_size(std::vector<ImageInfo> _info) {

    int min_X = INT_MAX;
    int min_Y = INT_MAX;
    int max_X = 0;
    int max_Y = 0;
    int C = -1;
    int W = -1;
    cv::Mat M = cv::Mat::eye(cv::Size(3, 3), CV_32F);

    for (std::vector<ImageInfo>::iterator itr = _info.begin();
        itr != _info.end(); ++itr) {

        cv::Mat Mc = itr->get_H();
        cv::Mat img = itr->get_img();

        int w = img.size().width;
        if (w == -1) {
            C = 3;
            W = w;
        }

        int _min_x, _min_y, _max_x, _max_y;

        this->set_bbox(img, Mc, _min_x, _min_y, _max_x, _max_y);

        min_X = std::min(min_X, _min_x);
        min_Y = std::min(min_Y, _min_y);
        max_X = std::max(max_X, _max_x);
        max_Y = std::max(max_Y, _max_y);
    }

    this->_H = int(std::ceil(max_Y) - std::floor(min_Y));
    this->_W = int(std::ceil(max_X) - std::floor(min_X));
    this->_C = C;

    float h_top_left[][3] = {1, 0, -min_X, 0, 1, -min_Y, 0, 0, 1};
    this->H_top_left = cv::Mat(cv::Size(3, 3), CV_32F, &h_top_left);
}


void Panorama::set_bbox(cv::Mat &img, cv::Mat &M,
    int &min_x, int &min_y, int &max_x, int &max_y) {

    cv::Size size = img.size();
    int h = size.height, w = size.width;

    float corners[][3] = {0, 0, 1, w, 0, 1, 0, h, 1, w, h, 1};
    cv::Mat corners_mat(cv::Size(3, 4), CV_32F, &corners);
    cv::Mat corners_transformed_mat = (M * corners_mat.t()).t();

    for (int i = 0; i < 4; ++i) {
        float weight = corners_transformed_mat.at<float>(i, 2);
        for (int j = 0; j < 2; ++j)
            corners_transformed_mat.at<float>(i, j) /= weight;
    }

    std::vector<float> X, Y;
    for (int i = 0; i < 4; ++i) {
        X.push_back(corners_transformed_mat.at<float>(i, 0));
        Y.push_back(corners_transformed_mat.at<float>(i, 1));
    }

    min_x = int(*std::min_element(std::begin(X), std::end(X)));
    min_y = int(*std::min_element(std::begin(Y), std::end(Y)));
    max_x = int(*std::max_element(std::begin(X), std::end(X)));
    max_y = int(*std::max_element(std::begin(Y), std::end(Y)));
}

void Panorama::paste_images(std::vector<ImageInfo> _info) {
    this->_final_panorama = cv::Mat::zeros(cv::Size(this->_W, this->_H), CV_32FC4);

    for (std::vector<ImageInfo>::iterator itr = _info.begin(); itr != _info.end(); ++itr) {
        cv::Mat M = itr->get_H();
        cv::Mat img = itr->get_img();

        M = this->H_top_left * M;
        // todo

    }
}

} // namespace reconstruct