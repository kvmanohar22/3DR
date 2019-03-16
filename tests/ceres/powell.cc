#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include <ceres/ceres.h>
#include <glog/logging.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// Powell's function
struct F1 {
   template <typename T>
   bool operator() (const T* const x1, const T* const x2, T* residual) const {
      residual[0] = T(x1[0] + T(10) * x2[0]);
      return true;
   }
};

struct F2 {
   template <typename T>
   bool operator() (const T* const x3, const T* const x4, T* residual) const {
      residual[0] = T(sqrt(5.0) * (x3[0] - x4[0]));
      return true;
   }
};

struct F3 {
   template <typename T>
   bool operator() (const T* const x2, const T* const x3, T* residual) const {
      residual[0] = T((x2[0] - T(2) * x3[0]) * (x2[0] - T(2) * x3[0]));
      return true;
   }
};

struct F4 {
   template <typename T>
   bool operator() (const T* const x1, const T* const x4, T* residual) const {
      residual[0] = T(sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]));
      return true;
   }
};


int main(int argc, char **argv) {
   using namespace std;
   using namespace ceres;

   google::InitGoogleLogging(argv[0]);

   double x1 = 3.0, x2 = -1.0, x3 = 0.0, x4 = 1.0;
   double x10 = x1, x20 = x2, x30 = x3, x40 = x4;

   Problem problem;
   problem.AddResidualBlock(
      new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x1, &x2);
   problem.AddResidualBlock(
      new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x3, &x4);
   problem.AddResidualBlock(
      new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x2, &x3);
   problem.AddResidualBlock(
      new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x1, &x4);

   Solver::Options opts;
   opts.linear_solver_type = ceres::DENSE_QR;
   opts.minimizer_progress_to_stdout = true;
   Solver::Summary summary;
   Solve(opts, &problem, &summary);

   cout << summary.FullReport() << endl;
   cout << "Initial values:\n"
        << "x1: " << x10 << " x2: " << x20 << " x2: " << x20 << " x2: " << x20 << endl
        << "Final values:\n"
        << "x1: " << x1 << " x2: " << x2 << " x2: " << x2 << " x2: " << x2 << endl;

   return 0;
}
