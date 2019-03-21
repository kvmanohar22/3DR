#include "point.hpp"

namespace dr3 {

Point::Point(cv::Mat xyz, size_t idx) {
   this->xyz = xyz.clone();
   this->idx = idx;
}

void Point::add_observation(Frame *frame,
                            unsigned int idx) {
   this->frames.push_back(frame);
   this->idxs.push_back(idx);
}

bool Point::is_valid() {
   return !xyz.empty();
}

cv::Mat Point::get_xyz() {
   return xyz.clone();
}

} // namespace dr3
