#ifndef _POINT_HPP_
#define _POINT_HPP_

#include "frame.hpp"

namespace dr3 {

class Point {
private:

   /* Position in the world

      Access the point as:
      x -> xyz.at<float>(0, 1);
      y -> xyz.at<float>(0, 2);
      z -> xyz.at<float>(0, 3);
   */
   cv::Mat xyz;

   // Frames observing this point and their indices
   std::vector<Frame*> frames;
   std::vector<unsigned int> idxs;

public:
   Point() {}
   Point(cv::Mat xyz);

   // Add a new observation
   void add_observation(Frame *frame, unsigned int idx);

   // Check if the point is valid
   bool is_valid();

   cv::Mat get_xyz();
}; // class Point

} // namespace dr3

#endif
