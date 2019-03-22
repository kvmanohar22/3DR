#include "point.hpp"

namespace dr3 {

Point::Point(cv::Mat xyz, unsigned int idx) {
   this->xyz = xyz.clone();
   this->idx = idx;
}

void Point::add_observation(Frame *frame,
                            unsigned int idx) {
   this->frames.push_back(frame);
   this->idxs.push_back(idx);
}

} // namespace dr3
