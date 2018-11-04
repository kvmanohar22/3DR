#ifndef _PANORAMA_HPP_
#define _PANORAMA_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include "dirent.h"

#include "utils.hpp"
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
        inline cv::Mat get_img() { return this->_img; }
        inline cv::Mat get_H() { return this->_H; }

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
        }

    ~Panorama() {}

    inline float get_focal_length() { return this->_focal_length; }
    inline float get_k1() { return this->_k1; }
    inline float get_k2() { return this->_k2; }
    inline int get_feathering_width() { return this->_feathering_width; }
    inline cv::Mat get_final_panorama() { return this->_final_panorama; }

    inline void load_images(char *dir_name) {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(dir_name)) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                this->_images.push_back(
                    utils::load_image(
                        std::string(ent->d_name)));
            }
            closedir(dir);
        }
        else {
            std::cerr << "Couldn't open the directory"
                      << dir_name << std::endl;
        }
    }

    int process(char *dir_name);
    void set_canvas_size(std::vector<ImageInfo> _info);
    void set_bbox(cv::Mat &img, cv::Mat &M,
        int &_min_x, int &_min_y, int &_max_x, int &_max_y);
    void paste_images(std::vector<ImageInfo> _info);


}; // Class Panorama

} // namespace reconstruct


#endif