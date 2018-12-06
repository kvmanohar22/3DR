#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

static const int W = 1920/2;
static const int H = 1080/2;
static const int MAX_CORNERS = 1000;

int process_frame(cv::Mat framein) {
  cv::Mat frame;
  cv::resize(framein, frame, cv::Size(W, H));

  // detect features in each image
  std::vector<cv::KeyPoint> kps;
  cv::Mat des, gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
  orb->detectAndCompute(gray, cv::noArray(), kps, des);

  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(gray, corners, MAX_CORNERS, 0.01, 3);

  for (auto itr : corners) {
    kps.push_back(cv::KeyPoint(itr, 20));
  }

  for (auto kp : kps) {
    cv::circle(frame, cv::Point(kp.pt.x, kp.pt.y), 2, cv::Scalar(0, 255, 0));
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
