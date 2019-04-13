#ifndef _3DR_FEATURE_DETECTION_HPP
#define _3DR_FEATURE_DETECTION_HPP

#include <config.hpp>
#include <frame.hpp>
#include <utils.hpp>

#include <vector>
#include <list>

#include <boost/shared_ptr.hpp>

#include <fast/fast.h>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace dr3 {

// Various feature detectors
namespace feature_detection {

using namespace Eigen;

// Image patch
class Feature {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // TODO: Add other types of features (eg: EDGELET, LINES etc)
    enum class FeatureType {
        CORNER
    };

    FeatureType type; // type of the corner
    Frame *frame;     // frame from which this feature was detected
    Vector2d px;      // coordinates in pixels (at pyramid level 0)
    Vector3d f;       // Unit-bearing vector of the feature
    int level;        // level from which this was extracted
    Point *point;     // 3D point in the world

    Feature(Frame *frame, const Vector2d &px, int level) :
        type(FeatureType::CORNER),
        frame(frame),
        px(px),
        f(0, 0, 0),
        level(level),
        point(NULL)
    {}
};
typedef std::list<Feature*> Features;

// Container for a corner in the image
class Corner {
public:
    int x;        // x-coordinate of the corner in the image
    int y;        // y-coordinate of the corner in the image
    int level;    // pyramid level of this corner
    float score;  // shi-tomasi score of the corner
    Corner(int x, int y, float score, int level) :
        x(x), y(y), level(level), score(score)
    {}
};
typedef std::vector<Corner> Corners;
typedef std::vector<cv::Mat> ImgPyramid;

/*
    - Base detector for all the feature detectors
    - All the feature detectors must derive from this
*/
class Detector {
public:
    Detector(
        const int img_w,
        const int img_h,
        const int cell_size,
        const int n_pyr_levels);

    virtual ~Detector() {};

    virtual void detect(
        Frame *frame,
        const ImgPyramid &img_pyr,
        const double detection_threshold,
        Features &fts) =0;

    // Flag the grid as occupied
    void flag_grid(const Vector2d &px);

    // Flag this grid cells of existing features as occupied
    void flag_features_grid(const Features &fts);

protected:
    static const int border = 8;
    const int cell_size;
    const int n_pyr_levels;
    const int grid_n_cols;
    const int grid_n_rows;
    std::vector<bool> grid_occupancy;

    void reset_grid();

    inline int get_cell_idx(int x, int y, int level) {
        const int scale = (1 << level);
        return (scale * y) / cell_size * grid_n_cols + (scale * x) / cell_size;
    }
};
typedef boost::shared_ptr<Detector> DetectorPtr;


class FastDetector : public Detector {
public:
    FastDetector(
        const int img_w,
        const int img_h,
        const int cell_size,
        const int n_pyr_levels);

    virtual ~FastDetector() {}

    virtual void detect(
        Frame *frame,
        const ImgPyramid &img_pyr,
        const double detection_threshold,
        Features &fts);
};


} // namespace feature_detection


} // namespace dr3

#endif // _3DR_FEATURE_DETECTION_HPP