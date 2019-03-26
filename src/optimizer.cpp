#include "optimizer.hpp"

namespace dr3 {

/*********************** OptProblem ***********************/
OptProblem::OptProblem(Map *map) : map(map) {

   std::vector<Frame*> frames = map->get_frames();
   std::vector<Point*> points = map->get_points();

   _num_cameras = frames.size();
   _num_points  = points.size();

   _num_observations = 0;
   for (auto &itr: points)
      _num_observations += itr->n_frames();

   // Note we are neglecting radial distortion
   _num_camera_parameters = 6 * _num_cameras;
   _num_point_parameters  = 3 * _num_points;

   // allocate memory
   _point_index  = new int[_num_observations];
   _camera_index = new int[_num_observations];
   _observations = new double[2 * _num_observations];
   _camera_parameters = new double[_num_camera_parameters];
   _point_parameters  = new double[_num_point_parameters];

   // Register observations
   size_t curr_observation_idx = 0;
   for (auto &itr: points) {
      std::vector<Frame*> frames = itr->get_frames();
      std::vector<unsigned int> indices = itr->get_indices();
      for (int i = 0; i < frames.size(); ++i, ++curr_observation_idx) {
         _camera_index[curr_observation_idx] = frames[i]->get_idx();
         _point_index[curr_observation_idx]  = itr->get_idx();
         cv::KeyPoint kpt = frames[i]->get_kpt(indices[i]);
         _observations[curr_observation_idx*2+0] = kpt.pt.x;
         _observations[curr_observation_idx*2+1] = kpt.pt.y;
      }
   }

   /* 
      Register camera parameters

      camera[0] -> rotation about x
      camera[1] -> rotation about y
      camera[2] -> rotation about z

      camera[3] -> translation about x
      camera[4] -> translation about y
      camera[5] -> translation about z
   */
   for (int i = 0; i < _num_cameras; ++i) {
      cv::Mat pose_w2c = frames[i]->get_pose_w2c();

      // Convert rotation matrix to angle-axis form
      double *angle_axis = new double[3];
      double *R = new double[9];
      for (int ii = 0; ii < 3; ++ii)
         for (int jj = 0; jj < 3; ++jj)
            R[ii*3+jj] = (double)pose_w2c.at<float>(ii, jj);
      RotationMatrixToAngleAxis(R, angle_axis);

      // Rotation
      for (int j = 0; j < 3; ++j)
         _camera_parameters[i*6+j] = angle_axis[j];

      // Translation
      for (int j = 0; j < 3; ++j)
         _camera_parameters[i*6+3+j] = (double)pose_w2c.at<float>(j, 3);
   } 

   // Register 3D point coordinates
   for (int i = 0; i < _num_points; ++i) {
      cv::Mat X3D = points[i]->get_xyz();
      for (int j = 0; j < 3; ++j) {
         _point_parameters[i*3+j] = (double)X3D.at<float>(j);
      }
   } 
}

void OptProblem::replace_back() {
   // put cameras back
   std::vector<Frame*> frames = map->get_frames();
   for (int i = 0; i < _num_cameras; ++i) {
      // Extract Rotation angles and translation vector
      double *curr_camera_params = this->mutable_cameras() + i * 6;
      double *angle_axis = new double[3];
      for (int j = 0; j < 3; ++j)
         angle_axis[j] = curr_camera_params[j];
      double *translation = new double[3];
      for (int j = 0; j < 3; ++j)
         translation[j] = curr_camera_params[3 + j];

      // Rotation angles -> Rotation matrix
      double *R = new double[9];
      AngleAxisToRotationMatrix(angle_axis, R);

      cv::Mat pose_w2c = cv::Mat::eye(4, 4, CV_32F);
      for (int ii = 0; ii < 3; ++ii)
         for (int jj = 0; jj < 3; ++jj)
            pose_w2c.at<float>(ii, jj) = (float)R[ii * 3 + jj];
      for (int ii = 0; ii < 3; ++ii)
         pose_w2c.at<float>(ii, 3) = (float)translation[ii];
      frames[i]->set_pose(pose_w2c);
   }

   // put 3D points back
   std::vector<Point*> points = map->get_points();
   for (int i = 0; i < _num_points; ++i) {
      double *curr_point_parameters = this->mutable_points() + i * 3;
      cv::Mat X3D(cv::Size(1, 3), CV_32F);
      for (int j = 0; j < 3; ++j) {
         X3D.at<float>(j) = (float)curr_point_parameters[j];
      }
      points[i]->set_xyz(X3D.clone());
   }
}

OptProblem::~OptProblem() {
   delete[] _point_index;
   delete[] _camera_index;
   delete[] _observations;
   delete[] _camera_parameters;
   delete[] _point_parameters;
}


/********************* Optimizer *********************/
void Optimizer::global_BA(Map *map, cv::Mat K) {
   OptProblem problem(map);

   const double* observations = problem.get_observations();

   double* intrinsics = new double[4];
   intrinsics[0] = (double)K.at<float>(0, 0);
   intrinsics[1] = (double)K.at<float>(1, 1);
   intrinsics[2] = (double)K.at<float>(0, 2);
   intrinsics[3] = (double)K.at<float>(1, 2);

   // Create residuals
   ceres::Problem ceres_problem;
   for (size_t i = 0; i < problem.num_observations(); ++i) {
      ceres::CostFunction *cost_function = 
         ReprojectionError::create(observations[2 * i + 0],
                                   observations[2 * i + 1]);
      ceres_problem.AddResidualBlock(cost_function,
                                     NULL,
                                     intrinsics,
                                     problem.mutable_camera_for_observation(i),
                                     problem.mutable_point_for_observation(i));
   }

   ceres::Solver::Options options;
   // options.linear_solver_type = ceres::SPARSE_SCHUR;        // Large BA
   // options.linear_solver_type = ceres::ITERATIVE_SCHUR;     // Thousands of cameras
   // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // If Jacobian has lot of zeros
   options.linear_solver_type = ceres::DENSE_SCHUR;            // Around of hundereds of cameras

   options.preconditioner_type = ceres::SCHUR_JACOBI;
   // options.preconditioner_type = ceres::CLUSTER_JACOBI;
   // options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL; 

   options.num_threads = 8;
   options.minimizer_progress_to_stdout = false;

   // Optimize
   ceres::Solver::Summary summary;
   ceres::Solve(options, &ceres_problem, &summary);
   std::cout << summary.FullReport() << std::endl;

   // Replace the optimized variables back
   problem.replace_back();
}


} // namespace dr3
