#include "panorama.hpp"

namespace reconstruct {

int Panorama::process(const char *dir_name) {

    // Load the images
    this->load_images(dir_name);
    std::vector<ImageInfo> _info;

    cv::Mat H_init = cv::Mat::eye(cv::Size(3, 3), CV_32F);
    ImageInfo img_info0(_images[0], H_init);
    _info.push_back(img_info0);

    for (int i = 0; i < this->_images.size()-1; ++i) {
        reconstruct::Stitch stitcher(_focal_length, _k1, _k2);
        int flag = stitcher.process(_images[i], _images[i+1]);

        cv::Mat H;
        if (flag == 0) {
            cv::Mat H_curr = stitcher.get_H();
            cv::Mat H_prev = _info[_info.size()-1].get_H();
            H = H_prev * H_curr;
        } else {
            std::cerr << "Couldn't process images: "
                      << i << i+1 << std::endl;
        }
        ImageInfo img_info(_images[i+1], H);
        _info.push_back(img_info);
    }

    // Compute the canvas size
    cv::Mat top_left = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
    this->set_canvas_size(_info, &top_left);

    // Paste images
    cv::Mat weighted_canvas = this->paste_images(_info, top_left);

    // Normalize the canvas
    this->normalize_canvas(weighted_canvas);

    return 0;
}

void Panorama::set_canvas_size(const std::vector<ImageInfo> &_info, cv::Mat *top_left) {

    int min_X = INT_MAX;
    int min_Y = INT_MAX;
    int max_X = 0;
    int max_Y = 0;
    int C = -1;
    int W = -1;
    cv::Mat M = cv::Mat::eye(cv::Size(3, 3), CV_32F);

    for (std::vector<ImageInfo>::const_iterator itr = _info.begin();
        itr != _info.end(); ++itr) {

        const cv::Mat Mc = itr->get_H();
        const cv::Mat img = itr->get_img();

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
    *top_left = cv::Mat(cv::Size(3, 3), CV_32F, &h_top_left).clone();

    std::cout << "Canvas H: " << _H << " " << " Canvas W: " << _W << std::endl;
}


void Panorama::set_bbox(const cv::Mat &img, const cv::Mat &M,
    int &min_x, int &min_y, int &max_x, int &max_y) {

    cv::Size size = img.size();
    int h = size.height, w = size.width;

    float corners[][3] = {0., 0., 1., w, 0., 1., 0., h, 1., w, h, 1.};
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

cv::Mat Panorama::paste_images(std::vector<ImageInfo> _info, cv::Mat top_left) {
    cv::Mat weighted_panorama = cv::Mat::zeros(cv::Size(this->_W, this->_H), CV_32FC4);

    for (auto &itr : _info) {
        cv::Mat M = itr.get_H();
        cv::Mat img = itr.get_img();
        M = top_left * M;
        this->add_img_to_canvas(img, M, this->get_feathering_width(), weighted_panorama);
    }
    return weighted_panorama;
}

void Panorama::add_img_to_canvas(cv::Mat img, cv::Mat M, int feathering_width,
    cv::Mat &weighted_panorama) {

    cv::Size size = img.size();
    int h = size.height, w = size.width;

    int min_x, min_y, max_x, max_y;
    this->set_bbox(img, M, min_x, min_y, max_x, max_y);

    int mid_len = max_x - min_x - 2 * this->_feathering_width;
    int temp_blendwidth = this->_feathering_width;
    if (mid_len < 0) {
        temp_blendwidth = (max_x - min_x) / 2 -1;
    }

    std::vector<float> left_blend  = utils::linspace<float>(0.f, 1.f, temp_blendwidth);
    std::vector<float> right_blend = utils::linspace<float>(1.f, 0.f, temp_blendwidth);
    std::vector<float> mid_blend(int(max_x - min_x - 2 * temp_blendwidth), 1.0f);

    left_blend.insert(left_blend.end(), mid_blend.begin(), mid_blend.end());
    left_blend.insert(left_blend.end(), right_blend.begin(), right_blend.end());

    cv::Mat weighted_img = cv::Mat::ones(cv::Size(w, h), CV_32FC4);
    cv::Mat img32f;
    img.convertTo(img32f, CV_32FC3);

    for (int ii = 0; ii < h; ++ii) {
        for (int jj = 0; jj < w; ++jj) {
            cv::Vec3f vv = img32f.at<cv::Vec3f>(ii, jj);
            // weighted_img.at<cv::Vec3f>(ii, jj)[0] = vv[0];
            // weighted_img.at<cv::Vec3f>(ii, jj)[1] = vv[1];
            // weighted_img.at<cv::Vec3f>(ii, jj)[2] = vv[2];
            weighted_img.at<cv::Vec4f>(ii, jj)[0] = vv[0];
            weighted_img.at<cv::Vec4f>(ii, jj)[1] = vv[1];
            weighted_img.at<cv::Vec4f>(ii, jj)[2] = vv[2];
        }
    }

    cv::Mat warped_img;
    cv::warpPerspective(weighted_img, warped_img, M, this->get_final_size(), CV_INTER_NN);

    for (int i = min_x; i != max_x; ++i) {
        for (int j = 0; j < warped_img.size().height; ++j) {
            // Weight each row of the current column
            warped_img.at<cv::Vec4f>(j, i)[0] *= left_blend[i-min_x];
            warped_img.at<cv::Vec4f>(j, i)[1] *= left_blend[i-min_x];
            warped_img.at<cv::Vec4f>(j, i)[2] *= left_blend[i-min_x];

            // set the weight
            warped_img.at<cv::Vec4f>(j, i)[3]  = left_blend[i-min_x];

            // Update the original canvas
            if (warped_img.at<cv::Vec4f>(j, i)[0] == 0 && warped_img.at<cv::Vec4f>(j, i)[1] == 0 && warped_img.at<cv::Vec4f>(j, i)[2] == 0)
                warped_img.at<cv::Vec4f>(j, i)[3] = 0.0f;
            weighted_panorama.at<cv::Vec4f>(j, i) += warped_img.at<cv::Vec4f>(j, i);
        }
    }
}

void Panorama::normalize_canvas(cv::Mat &weighted_panorama) {
    this->_final_panorama = cv::Mat::zeros(cv::Size(this->_W, this->_H), CV_8UC3);
    for (int i = 0; i < this->_H; ++i) {
        for (int j = 0; j < this->_W; ++j) {
            float alpha_sum = weighted_panorama.at<cv::Vec4f>(i, j)[3];
            if (alpha_sum > 0.0f) {
                // std::cout << "here " << std::endl;
                cv::Vec3b vec;
                for (int c = 0; c < 3; ++c) {
                    vec[c] = (int)weighted_panorama.at<cv::Vec4f>(i, j)[c] / alpha_sum;
                }
                this->_final_panorama.at<cv::Vec3b>(i, j) = vec;
            } else {
                // std::cout << "all zero?" << std::endl;
            }
        }
    }
}

} // namespace reconstruct
