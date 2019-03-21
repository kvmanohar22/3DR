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

   inline const long unsigned int get_idx() const { return idx; }
   inline const std::vector<cv::KeyPoint> get_kps() const { return kps; }
   inline const cv::Mat get_des() const { return des; }

   // camera poses
   inline cv::Mat get_pose_w2c() const { return pose_w2c.clone(); }
   inline cv::Mat get_pose_c2w() const { return pose_c2w.clone(); }
   inline cv::Mat get_center()   const { return center.clone();   }
   
   // Set the pose of the matrix (world -> camera)
   void set_pose(cv::Mat pose);
   void update_poses();

   // Compute keypoints of the given image
   void compute_kps(const cv::Mat &img);
}; // class Frame

} // namespace dr3

#endif
