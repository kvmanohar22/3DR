#ifndef _GLOBAL_HPP_
#define _GLOBAL_HPP_

#include <vector>
#include <list>
#include <algorithm>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <Eigen/StdVector>

#include <sophus/se3.h>


EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d)

namespace dr3 {

    using namespace std;
    using namespace Eigen;
    using namespace Sophus;

    class Frame;
    class Feature;
    typedef boost::shared_ptr<Frame> FramePtr;
    typedef std::vector<cv::Mat> ImgPyramid;

} // namespace dr3

#endif // _GLOBAL_HPP_
