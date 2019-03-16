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


// Compute Jacobians manually (Not a good idea)
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
public:
   virtual ~QuadraticCostFunction() {}
   virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const {
      const double x = parameters[0][0];
      residuals[0] = 10.0f - x;

      if (jacobians != NULL && jacobians[0] != NULL) {
         jacobians[0][0] = -1;
      }
      return true;
   }
};

// For Automatic Differentiation
struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T *residual) const {
      residual[0] = T(10.0) - x[0];
      return true;
   }
};


int main(int argc, char **argv) {
   using namespace std;
   using namespace ceres;

   google::InitGoogleLogging(argv[0]);

   double initial_x = 5.0f;
   double x = initial_x;

   Problem problem;

   CostFunction *cost_function = 
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
   problem.AddResidualBlock(cost_function, NULL, &x);

   Solver::Options opts;
   opts.linear_solver_type = ceres::DENSE_QR;
   opts.minimizer_progress_to_stdout = true;
   Solver::Summary summary;
   Solve(opts, &problem, &summary);

   cout << summary.BriefReport() << endl;
   cout << "x: " << initial_x
        << " -> " << x << endl;

   return 0;
}
