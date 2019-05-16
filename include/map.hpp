#ifndef _MAP_HPP_
#define _MAP_HPP_

#include "point.hpp"
#include "frame.hpp"

#include <vector>
#include <set>
#include <list>

namespace dr3 {

class Map {
protected:
   list<Point*>   _points; // All the converged points in the map
   list<FramePtr> _frames; // All the keyframes

public:
   Map() =default;

   void add_point(Point *point);
   void add_frame(FramePtr frame);

   vector<FramePtr> get_frames();
   vector<Point*> get_points();

   size_t n_frames() { return _frames.size(); }
   size_t n_points() { return _points.size(); }

   size_t n_observations();
}; // class Map

} // namespace dr3

#endif
