#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <fstream>

#include "slam.hpp"

using namespace std;

void load_images(const string sequence_root_path,
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

   string dir = sequence_root_path + "/image_0/";

   const int times = time_stamps_data.size();
   imgs.resize(times);

   for(int i = 0; i < times; i++) {
      stringstream ss;
      ss << setfill('0') << setw(6) << i;
      imgs[i] = dir + ss.str() + ".png";
   }
}

int main(int argc, char **argv) {

   if (argc != 2) {
      cerr << "Usage: ./stereo path_to_sequence_root" << endl;
   }

   // image dimensions
   const int H = 376;
   const int W = 1240;

   // camera intrinsics
   float bf = 386.1448;
   float fx = 718.8560;
   float fy = 718.8560;
   float cx = 607.1928;
   float cy = 185.2157;
   cv::Mat K = (cv::Mat_<float>(3, 3) << fx,  0, cx,
                                          0, fy, cy,
                                          0,  0,  1);
   dr3::SLAM slam(H, W, K);

   // load image paths
   vector<string> imgs;
   load_images(argv[1], imgs);
   const int n_images = imgs.size();

   // process
   cv::Mat img;
   for (int i = 0; i < n_images; ++i) {
      img = cv::imread(imgs[i], CV_LOAD_IMAGE_UNCHANGED);
      slam.process(img);
   }
}
