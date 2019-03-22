#ifndef _POINT_HPP_
#define _POINT_HPP_

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "frame.hpp"

namespace dr3 {

class Frame;

class Point {
private:

   // unique point idx
   size_t idx;

   /* Position in the world

      Access the point as:
      x -> xyz.at<float>(0);
      y -> xyz.at<float>(1);
      z -> xyz.at<float>(2);
   */
   cv::Mat xyz;

   // Frames observing this point and their (keypoint) indices
   std::vector<Frame*> frames;
   std::vector<unsigned int> idxs;

public:
   Point() {}
   Point(cv::Mat xyz, size_t idx);

   inline size_t get_idx() { return idx; }

   // Add a new observation
   void add_observation(Frame *frame, unsigned int idx);

   // Check if the point is valid
   bool is_valid() { return !xyz.empty(); }

   cv::Mat get_xyz() { return xyz.clone(); }
}; // class Point

} // namespace dr3

#endif
