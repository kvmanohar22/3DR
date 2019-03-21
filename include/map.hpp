#ifndef _MAP_HPP_
#define _MAP_HPP_

#include "point.hpp"
#include "frame.hpp"

#include <vector>
#include <set>

namespace dr3 {

class Map {
protected:
   std::vector<Point*> points;
   std::vector<Frame*> frames;

public:
   Map() {}

   void add_point(Point *point);
   void add_frame(Frame *frame);

   std::vector<Frame*> get_frames();
   std::vector<Point*> get_points();

   size_t n_frames() { return frames.size(); }
   size_t n_points() { return points.size(); }
}; // class Map

} // namespace dr3

#endif
