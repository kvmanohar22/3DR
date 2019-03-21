#include "optimizer.hpp"

namespace dr3 {

Optimizer::Optimizer() {
   _options.linear_solver_type = ceres::DENSE_SCHUR;
   _options.minimizer_progress_to_stdout = true;
}

void Optimizer::global_BA(Map *map) {

}


} // namespace dr3
