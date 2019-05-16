#include "map.hpp"

namespace dr3 {

void Map::add_point(Point *point) {
   this->_points.push_back(point);
}

void Map::add_frame(const FramePtr frame) {
   this->_frames.push_back(frame);
}

vector<FramePtr> Map::get_frames() {
   return vector<FramePtr>(_frames.begin(), _frames.end());
}

vector<Point*> Map::get_points() {
   return vector<Point*>(_points.begin(), _points.end());
}

size_t Map::n_observations() {
  size_t n_obsers = 0;
  for (auto &itr: _points)
    n_obsers += itr->n_frames();
  return n_obsers;
}

} // namespace dr3
