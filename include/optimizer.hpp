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
#include "point.hpp"
#include "frame.hpp"

namespace dr3 {

class Map;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace ceres;

/*********************** OptProblem ***********************/
class OptProblem {
private:
   int _num_cameras;
   int _num_points;
   int _num_observations;
   int _num_camera_parameters;
   int _num_point_parameters;

   int *_point_index;
   int *_camera_index;
   float *_observations;
   float *_camera_parameters;
   float *_point_parameters;

   Map *mapp;

public:
   OptProblem(Map *mapp);
   ~OptProblem();

   inline int num_observations() { return _num_observations; }
   inline int num_points() { return _num_points; }
   inline int num_cameras() { return _num_cameras; }
   inline int num_camera_parameters() { return _num_camera_parameters; }
   inline int num_point_parameters() { return _num_point_parameters; }
};


/********************* Reprojection Error *********************/
struct ReprojectionError {
   float observed_x;
   float observed_y;

   ReprojectionError(float observed_x, float observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

   template <typename T>
   bool operator() (const T* const intrinsics,
                    const T* const camera,
                    const T* const point,
                    T* residuals) const {

      // Extract intrinsics
      const T fx = intrinsics[0];
      const T fy = intrinsics[1];
      const T px = intrinsics[2];
      const T py = intrinsics[3];

      // Rotate
      T p[3];
      ceres::AngleAxisRotatePoint(camera, point, p);

      // Translate
      p[0] += camera[3];
      p[1] += camera[4];
      p[2] += camera[5];

      // convert to image coordinates
      T predicted_x = fx * p[0] / p[2] + px;
      T predicted_y = fy * p[1] / p[2] + py;

      // Error
      residuals[0] = predicted_x - T(observed_x);
      residuals[1] = predicted_y - T(observed_y);

      return true;      
   }

   // Create loss
   static ceres::CostFunction* create(const float observed_x,
                                      const float observed_y) {
      return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 6, 3>(
                  new ReprojectionError(observed_x, observed_y)));
   }
};


/********************* Optimizer *********************/
class Optimizer {
private:

   // solver
   static Solver::Options _options;
   static Solver::Summary _summary;

public:   
   Optimizer();

   static void global_BA(Map *map);

}; // class Optimizer

} // namespace 3dr 

#endif

