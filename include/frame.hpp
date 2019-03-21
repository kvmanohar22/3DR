#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

namespace dr3 {

class Frame {
private:
   // Unique frame index
   long unsigned int idx;

   // intrinsics
   cv::Mat K;

   // Keypoints and descriptors
   std::vector<cv::KeyPoint> kps;
   cv::Mat des;

   // Camera pose
   cv::Mat pose_w2c; // world  -> camera
   cv::Mat pose_c2w; // camera -> world
   cv::Mat center;   // camera center in world coordinates


public:
   Frame();
   Frame(Frame &frame);
   Frame(const long unsigned int idx, 
         const cv::Mat &img, const cv::Mat &K);

   long unsigned int get_idx() const { return idx; }
   std::vector<cv::KeyPoint> get_kps() const { return kps; }
   cv::Mat get_des() const { return des; }
   cv::Mat get_pose(bool _short) const;
   cv::Mat get_center(bool _short) const;
   
   // Set the pose of the matrix
   void set_pose(cv::Mat pose);
   void update_poses();

   // Compute keypoints of the given image
   void compute_kps(const cv::Mat &img);
}; // class Frame

} // namespace dr3

#endif
