#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <glog/logging.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;
using namespace ceres;

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
   public:
   ~BALProblem() {
      delete[] point_index_;
      delete[] camera_index_;
      delete[] observations_;
      delete[] parameters_;
   }

   int num_observations()       const { return num_observations_;               }
   const double* observations() const { return observations_;                   }
   double* mutable_cameras()          { return parameters_;                     }
   double* mutable_points()           { return parameters_  + 9 * num_cameras_; }

   double* mutable_camera_for_observation(int i) {
      return mutable_cameras() + camera_index_[i] * 9;
   }

   double* mutable_point_for_observation(int i) {
      return mutable_points() + point_index_[i] * 3;
   }

   bool LoadFile(const char* filename) {
      FILE* fptr = fopen(filename, "r");
      if (fptr == NULL) {
         return false;
      } else {
         cout << "File: " << filename << " opened successfully\n";
      }

      FscanfOrDie(fptr, "%d", &num_cameras_);
      FscanfOrDie(fptr, "%d", &num_points_);
      FscanfOrDie(fptr, "%d", &num_observations_);

      point_index_ = new int[num_observations_];
      camera_index_ = new int[num_observations_];
      observations_ = new double[2 * num_observations_];
      num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
      parameters_ = new double[num_parameters_];

      for (int i = 0; i < num_observations_; ++i) {
         FscanfOrDie(fptr, "%d", camera_index_ + i);
         FscanfOrDie(fptr, "%d", point_index_ + i);
         for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
         }
      }
      for (int i = 0; i < num_parameters_; ++i) {
         FscanfOrDie(fptr, "%lf", parameters_ + i);
      }
      return true;
   }

   private:
   template<typename T>
   void FscanfOrDie(FILE *fptr, const char *format, T *value) {
      int num_scanned = fscanf(fptr, format, value);
      if (num_scanned != 1) {
         LOG(FATAL) << "Invalid UW data file.";
      }
   }
   int num_cameras_;
   int num_points_;
   int num_observations_;
   int num_parameters_;
   int* point_index_;
   int* camera_index_;
   double* observations_;
   double* parameters_;
};


struct ReprojectionError {
   ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

   template <typename T>
   bool operator() (const T* const camera,
                    const T* const point,
                    T *residual) const {
      T p[3];
      ceres::AngleAxisRotatePoint(camera, point, p);

      p[0] += camera[3];
      p[1] += camera[4];
      p[2] += camera[5];

      T xp = -p[0] / p[2];
      T yp = -p[1] / p[2];

      const T &l1 = camera[7];
      const T &l2 = camera[8];
      T r2 = xp*xp + yp*yp;
      T distortion = T(1.0) + r2 * (l1 + l2 * r2);

      const T &focal = camera[6];
      T pred_x = focal * distortion * xp;
      T pred_y = focal * distortion * yp;

      residual[0] = pred_x - T(observed_x);
      residual[1] = pred_y - T(observed_y);

      return true;
   }

   static ceres::CostFunction* create(const double obs_x,
                                      const double obs_y) {
      return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 3>(
                  new ReprojectionError(obs_x, obs_y)));
   }

   double observed_x;
   double observed_y;
};


int main(int argc, char **argv) {
   google::InitGoogleLogging(argv[0]);

   BALProblem bal_problem;
   if (!bal_problem.LoadFile(argv[1])) {
      cerr << "usage: ba <bal_problem>\n";
      return 1;
   }
   const double *observations = bal_problem.observations();

   Problem problem;
   for (int i = 0; i < 1; ++i) {
      ceres::CostFunction *cost_func = 
         ReprojectionError::create(
            observations[2 * i + 0],
            observations[2 * i + 1]);
      problem.AddResidualBlock(cost_func,
                               NULL,
                               bal_problem.mutable_camera_for_observation(i),
                               bal_problem.mutable_point_for_observation(i));
   }

   Solver::Options opts;
   opts.linear_solver_type = ceres::DENSE_SCHUR;
   opts.minimizer_progress_to_stdout = true;

   Solver::Summary summary;
   Solve(opts, &problem, &summary);
   cout << summary.FullReport() << endl;
   return 0;
}
