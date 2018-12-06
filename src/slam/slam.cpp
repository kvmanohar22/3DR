#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

#include "slam/utils.hpp"

static const int W = 1920/2;
static const int H = 1080/2;
static const int MAX_CORNERS = 3000;
static const int F = 400;

using namespace std;
std::vector<utils::Frame> frames;

int process_frame(cv::Mat framein) {
  cv::Mat frame;
  cv::resize(framein, frame, cv::Size(W, H));

  utils::Frame fr(frame);
  std::vector<cv::KeyPoint> kps = fr.get_kps();
  frames.push_back(fr);

  if (frames.size() < 2)
    return 0;
  
  utils::Frame f1 = frames[frames.size()-1];
  utils::Frame f2 = frames[frames.size()-2];

  std::vector<int> idx1, idx2;
  utils::match_frames(f1, f2, idx1, idx2);

  std::vector<uchar> inliers(idx1.size(), 0);
  cv::Mat fmat = utils::estimate_fundamental_matrix(f1, f2, idx1, idx2, inliers);

  size_t last = 0;
  for (size_t i = 0; i < idx1.size(); i++) {
    if (inliers[i]) {
      idx1[last] = idx1[i];
      idx2[last] = idx2[i];
      ++last;
    }
  }
  idx1.erase(idx1.begin() + last, idx1.end());
  idx2.erase(idx2.begin() + last, idx2.end());

  // extract R,t matrices
  cv::Mat Rt(cv::Size(4, 3), CV_32F);
  Rt = utils::extractRt(fmat);
  f1.pose = Rt * f2.pose;
  exit(-2);



  for (int i = 0; i < idx1.size(); ++i) {
    int ii = idx1[i];
    int jj = idx2[i];
    cv::Point p1 = cv::Point(f1.get_kps()[ii].pt.x, f1.get_kps()[ii].pt.y);
    cv::Point p2 = cv::Point(f2.get_kps()[jj].pt.x, f2.get_kps()[jj].pt.y);
    
    cv::circle(frame, p1, 2, cv::Scalar(0, 255, 0));
    cv::circle(frame, p2, 2, cv::Scalar(0, 0, 255));
    cv::line(frame, p1, p2, cv::Scalar(255, 0, 0));
  }

  cv::imshow("SLAM", frame);
  cv::waitKey(20);
}

int main() {
  const std::string vid_file("../videos/test.mp4");
  cv::VideoCapture capture;
  capture.open(vid_file);
  if (!capture.isOpened()) {
    std::cerr << "Couldn't open video file: " << vid_file << std::endl;
    exit(-1);
  }

  cv::Mat frame;
  cv::namedWindow("SLAM", 1);
  int i = 0;
  while (true) {
    capture >> frame;
    if (frame.empty())
      break;
    process_frame(frame);
  }
  cv::waitKey(0);
}
