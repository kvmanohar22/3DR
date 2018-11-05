#ifndef _PANORAMA_HPP_
#define _PANORAMA_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include "dirent.h"

#include "stitch.hpp"

namespace reconstruct {

class ImageInfo {
    private:
        // Image
        cv::Mat _img;

        // Homography wrt first image
        // (in global co-ordinates of final image frame)
        cv::Mat _H;

    public:
        ImageInfo() {}
        ImageInfo(cv::Mat _img, cv::Mat _H) {
            this->_img = _img;
            this->_H   = _H;
        }

        ~ImageInfo() {}

        inline const cv::Mat get_img() const { return this->_img; }
        inline const cv::Mat get_H() const { return this->_H; }

        inline void set_img(cv::Mat _img) { this->_img = _img; }
        inline void set_H(cv::Mat _H) { this->_H = _H; }
}; // class ImageInfo

class Panorama {

private:
    // input images
    std::vector<cv::Mat> _images;

    // camera parameters
    float _focal_length;
    float _k1;
    float _k2;

    // feathering width while blending
    int _feathering_width;

    // final canvas dimensions
    int _W, _H, _C;

    // Homography to translate the first image to left most position
    cv::Mat H_top_left;

    // final panorama
    cv::Mat _final_panorama;

public:
    Panorama() {
        this->_focal_length = utils::INF;
        this->_k1 = utils::INF;
        this->_k2 = utils::INF;
        this->_feathering_width = utils::INF;
    }

    Panorama(float _focal_length, float _k1, float _k2, 
        int _feathering_width) {
            this->_focal_length = _focal_length;
            this->_k1 = _k1;
            this->_k2 = _k2;
            this->_feathering_width = _feathering_width;
            this->H_top_left = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
        }

    ~Panorama() {}

    inline float get_focal_length() { return this->_focal_length; }
    inline float get_k1() { return this->_k1; }
    inline float get_k2() { return this->_k2; }
    inline int get_feathering_width() { return this->_feathering_width; }
    inline cv::Mat get_final_panorama() { return this->_final_panorama; }
    inline cv::Size get_final_size() { return cv::Size(this->_W, this->_H); }

    inline void set_top_left(cv::Mat H_top_left) { this->H_top_left = H_top_left; }
    inline cv::Mat get_top_left() const { return this->H_top_left; }

    inline void load_images(const char *dir_name) {
        std::vector<std::string> files;
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(dir_name)) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                if (std::string(ent->d_name) == "." || std::string(ent->d_name) == "..")
                    continue;
                std::string img_file = std::string(dir_name)+std::string("/")+std::string(ent->d_name);
                files.push_back(img_file);
            }
            closedir(dir);
        }
        else {
            std::cerr << "Couldn't open the directory"
                      << dir_name << std::endl;
        }

        std::cout << "Found a total of " << files.size() << " images" << std::endl;
        std::sort(files.begin(), files.end());
        for (auto &file : files)
            this->_images.push_back(utils::load_image(file));
    }

    int process(const char *dir_name);
    void set_canvas_size(const std::vector<ImageInfo> &_info, cv::Mat *h_top_left);
    void set_bbox(const cv::Mat &img, const cv::Mat &M,
        int &_min_x, int &_min_y, int &_max_x, int &_max_y);
    cv::Mat paste_images(std::vector<ImageInfo> _info, cv::Mat top_left);
    void add_img_to_canvas(cv::Mat img, cv::Mat M, int feathering_width, 
        cv::Mat &weighted_panorama);
    void normalize_canvas(cv::Mat &weighted_panorama);

}; // Class Panorama

} // namespace reconstruct


#endif