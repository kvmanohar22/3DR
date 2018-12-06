#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>

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

  // Extract homogenous coordinates (scene points)
  std::vector<cv::Point2f> kps1, kps2;
  std::vector<cv::KeyPoint> kpa1 = f1.get_kps();
  std::vector<cv::KeyPoint> kpa2 = f2.get_kps();
  for (auto i : idx1) {
    float x = kpa1[i].pt.x;
    float y = kpa1[i].pt.y;
    kps1.push_back(cv::Point2f(x, y));
  }
  for (auto i : idx2) {
    float x = kpa2[i].pt.x;
    float y = kpa2[i].pt.y;
    kps2.push_back(cv::Point2f(x, y));
  }
  cv::Mat pts4d = cv::Mat::zeros(cv::Size(4, kps1.size()), CV_64FC1);
  cv::Mat f1pose = cv::Mat::zeros(cv::Size(4, 3), CV_64F);
  cv::Mat f2pose = cv::Mat::zeros(cv::Size(4, 3), CV_64F);
  for (int i = 0; i < 3; ++i) {
      f1.pose.row(i).copyTo(f1pose.row(i));
      f2.pose.row(i).copyTo(f2pose.row(i));
    }
  cv::triangulatePoints(f1pose, f2pose, kps1, kps2, pts4d);
  for (int i = 0; i < pts4d.cols; ++i) {
    double w = pts4d.at<double>(3, i);
    for (int j = 0; j < 3; ++j)
      pts4d.at<double>(j, i) /= w;
  }
  std::vector<int> good_pts(pts4d.cols, 0);
  for (int i = 0; i < pts4d.cols; ++i)
    if (std::abs(pts4d.at<double>(3, i)) > 0.005 && pts4d.at<double>(2, i) > 0)
      good_pts[i] = 1;
  exit(1);

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
  while (true) {
    capture >> frame;
    if (frame.empty())
      break;
    process_frame(frame);
  }
  cv::waitKey(0);
}
