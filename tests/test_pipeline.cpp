//
// Created by kv on 16/5/19.
//

#include "svo/handler.h"
#include "features.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "viewer.hpp"
#include "camera.hpp"
#include "svo/initialization.hpp"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <iomanip>

using namespace dr3;
using namespace std;

void load_images(const string &sequence_root_path,
                 vector<string> &imgs) {
    ifstream time_stamps;
    string time_stamp_file = sequence_root_path + "/times.txt";
    time_stamps.open(time_stamp_file.c_str());
    vector<double> time_stamps_data;
    while(!time_stamps.eof()){
        string s;
        getline(time_stamps, s);

        if(!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            time_stamps_data.push_back(t);
        }
    }

    string dir = sequence_root_path + "/img/";

    const int times = time_stamps_data.size();
    imgs.resize(times);

    for(int i = 0; i < times; i++) {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        imgs[i] = dir + ss.str() + ".png";
    }
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 2) {
        LOG(FATAL) << "Usage: ./test_pipeline path_to_sequence";
    }

    // camera
    AbstractCamera *cam = new Pinhole(1240, 376,
                                      718.856, 718.856,
                                      607.1928, 185.2157);
    HandlerMono _handler(cam);

    vector<string> img_files;
    load_images(argv[1], img_files);

    cv::Mat img;
    for (auto const &img_file: img_files) {
        img = cv::imread(img_file, CV_LOAD_IMAGE_UNCHANGED);
        _handler.add_image(img, 0.0);
    }
}
