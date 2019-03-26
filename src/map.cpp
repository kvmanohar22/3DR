#include "map.hpp"

namespace dr3 {

void Map::add_point(Point *point) {
   this->points.push_back(point);
}

void Map::add_frame(Frame *frame) {
   this->frames.push_back(frame);
}

std::vector<Frame*> Map::get_frames() {
   return std::vector<Frame*>(frames.begin(), frames.end());
}

std::vector<Point*> Map::get_points() {
   return std::vector<Point*>(points.begin(), points.end());
}

size_t Map::n_observations() {
  size_t n_obsers = 0;
  for (auto &itr: points)
    n_obsers += itr->n_frames();
  return n_obsers;
}

} // namespace dr3
