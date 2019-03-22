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
   unsigned int idx;

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
   Point(cv::Mat xyz, unsigned int idx);

   inline unsigned int get_idx() { return idx; }

   // Add a new observation
   void add_observation(Frame *frame, unsigned int idx);

   inline bool is_valid() { return !xyz.empty(); }
   inline cv::Mat get_xyz() { return xyz.clone(); }
   inline size_t n_frames() const { return frames.size(); }
   inline std::vector<Frame*> get_frames() { return frames; }
   inline std::vector<unsigned int> get_indices() { return idxs; }

}; // class Point

} // namespace dr3

#endif
