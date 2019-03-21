#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <glog/logging.h>

#include "map.hpp"

namespace dr3 {

class Map;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace ceres;

enum class OptType {
   LOCAL,
   GLOBAL,
   MOTION_ONLY
};


class Optimizer {
private:
   OptType _type;

   // solver
   Solver::Options _options;
   Solver::Summary _summary;


public:   
   Optimizer();

   void global_BA(Map *map);

}; // class Optimizer

} // namespace 3dr 

#endif

