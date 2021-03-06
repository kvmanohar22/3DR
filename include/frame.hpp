#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "point.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "camera.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

namespace dr3 {

typedef std::vector<cv::Mat> ImgPyramid;
typedef std::list<Feature*> Features;

class Point;

class Frame {
public:
    // Unique frame index
    long unsigned int idx;

    // Redundant as of now (remove)
    cv::Mat K;
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;

    AbstractCamera*  _cam;         // camera
    ImgPyramid       _img_pyr;     // image pyramid
    Features         _fts;         // frame features
    Sophus::SE3      _T_f_w;       // world frame -> camera frame
    bool             _is_keyframe; // is frame this keyframe?
    double           _time_stamp;  // Timestamp of when the image was acquired

    /// Will be removed in the future
    cv::Mat          pose_w2c; // world  -> camera
    cv::Mat          pose_c2w; // camera -> world
    cv::Mat          center;   // camera center in world coordinates

    // Points projected from this Frame onto 3D world
    std::vector<Point*> points;

    Frame() =default;
    Frame(Frame &frame);
    Frame(const long unsigned int idx,
         const cv::Mat &img, AbstractCamera *cam,
         double time_stamp);

    inline const long unsigned int get_idx() const { return idx; }
    inline const std::vector<cv::KeyPoint> get_kps() const { return kps; }
    inline cv::KeyPoint get_kpt(size_t idx) const { return kps[idx]; }
    inline const cv::Mat get_des() const { return des; }

    // camera poses
    inline cv::Mat get_pose_w2c() const { return pose_w2c.clone(); }
    inline cv::Mat get_pose_c2w() const { return pose_c2w.clone(); }
    inline cv::Mat get_center()   const { return center.clone();   }

    // Set the pose of the matrix (world -> camera)
    void set_pose(cv::Mat pose);
    void update_poses();

    // Compute keypoints of the given image
    void compute_kps(const cv::Mat &img);

    // add observation (point corresponds to kps[idx])
    void add_observation(Point *point, size_t idx);

    // Checks if a 3D point at `idx` is not back projected
    bool fresh_point(size_t idx) { return points[idx] == NULL; }

    Point* get_point(size_t idx) { return points[idx]; }

    /// Camera instance
    inline AbstractCamera* get_cam() { return _cam; }

    /// Return the pose of the frame in the (w)orld coordinate frame.
    inline Vector3d pos() const { return _T_f_w.inverse().translation(); }

    /// Compute features
    bool compute_features(Features &features);

    /// Add a new observation
    void add_observation(Feature *feature);

    /// Is the current frame a keyframe?
    void set_keyframe() { _is_keyframe = true; }

    bool is_keyframe() { return _is_keyframe; }

}; // class Frame

} // namespace dr3

#endif
