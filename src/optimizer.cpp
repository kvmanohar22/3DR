#include "optimizer.hpp"

namespace dr3 {

/*********************** OptProblem ***********************/
OptProblem::OptProblem(Map *mapp) {

   std::vector<Frame*> frames = mapp->get_frames();
   std::vector<Point*> points = mapp->get_points();

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
   _observations = new float[2 * _num_observations];
   _camera_parameters = new float[_num_camera_parameters];
   _point_parameters  = new float[_num_point_parameters];

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
      float *angle_axis = new float[3];
      float *R = new float[9];
      for (int ii = 0; ii < 3; ++ii)
         for (int jj = 0; jj < 3; ++jj)
            R[ii*3+jj] = pose_w2c.at<float>(ii, jj);
      RotationMatrixToAngleAxis(R, angle_axis);

      // Rotation
      for (int j = 0; j < 3; ++j)
         _camera_parameters[i*6+j] = angle_axis[j];

      // Translation
      for (int j = 0; j < 3; ++j)
         _camera_parameters[i*6+3+j] = pose_w2c.at<float>(j, 3);
   } 

   // Register 3D point coordinates
   for (int i = 0; i < _num_points; ++i) {
      cv::Mat X3D = points[i]->get_xyz();
      for (int j = 0; j < 3; ++j) {
         _point_parameters[i*3+j] = X3D.at<float>(j);
      }
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

   const float* observations = problem.get_observations();

   float* intrinsics = new float[4];
   intrinsics[0] = K.at<float>(0, 0);
   intrinsics[1] = K.at<float>(1, 1);
   intrinsics[2] = K.at<float>(0, 2);
   intrinsics[3] = K.at<float>(1, 2);

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
   options.linear_solver_type = ceres::DENSE_SCHUR;
   options.minimizer_progress_to_stdout = true;

   ceres::Solver::Summary summary;
   ceres::Solve(options, &ceres_problem, &summary);
}


} // namespace dr3
