#ifndef _INITIALIZATION_HPP_
#define _INITIALIZATION_HPP_

#include "global.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "features.hpp"
#include "viewer.hpp"

#include <vector>
#include <sys/time.h>

namespace dr3 {

namespace init {

enum class Result {
    FAILED,
    SUCCESS
};

typedef pair<int, int> Match;

class InitHelper {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Fix the reference frame
    InitHelper(const FramePtr &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(const FramePtr &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

    inline vector<cv::KeyPoint>& mutable_keys_ref() { return mvKeys1; }
    inline vector<cv::KeyPoint>& mutable_keys_cur() { return mvKeys2; }

private:
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);
    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                const vector<Match> &vMatches12, vector<bool> &vbInliers,
                const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

    // Keypoints from Reference Frame (Frame 1)
    vector<cv::KeyPoint> mvKeys1;

    // Keypoints from Current Frame (Frame 2)
    vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Reference to Current
    vector<Match> mvMatches12;
    vector<bool> mvbMatched1;

    // Calibration
    cv::Mat mK;

    // Standard Deviation and Variance
    float mSigma, mSigma2;

    // Ransac max iterations
    int mMaxIterations;

    // Ransac sets
    vector<vector<size_t> > mvSets;
};

class Init {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FramePtr _frame_ref;
    Init() =default;
    ~Init() =default;

    Result add_first_frame(FramePtr frame_ref);

    // This is outdated method (only fucking works for downward facing cameras)
    Result add_second_frame(FramePtr frame_cur);
    double compute_inliers(const Matrix3d &R,
                           const Vector3d &t);

    // More generalized method
    Result add_second_frame_generalized(FramePtr frame_cur);

protected:
    vector<cv::Point2f> _kps_ref;        // keypoints in the ref frame
    vector<cv::Point2f> _kps_cur;        // keypoints in the cur frame
    vector<Vector3d>    _pts_ref;        // bearing vectors in the ref frame
    vector<Vector3d>    _pts_cur;        // bearing vectors in the cur frame
    vector<Vector3d>    _xyz_in_cur;     // 3D points after homography
    vector<double>      _disparities;    // Disparities for each matching point
    vector<int>         _inliers;        // inlier indices
    SE3                 _T_cur_from_ref; // Transformation matrix (ref -> cur)
};


class InitMain {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    InitMain();

    // Initialization Variables
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;


    InitHelper *initializer;
    FramePtr   frame_ref;
    FramePtr   frame_cur;

    vector<cv::KeyPoint> kpts_ref;
    vector<cv::KeyPoint> kpts_cur;

    Result process(FramePtr &frame);

protected:
    vector<cv::Point2f> _kps_ref;        // keypoints in the ref frame
    vector<cv::Point2f> _kps_cur;        // keypoints in the cur frame
    vector<Vector3d>    _pts_ref;        // bearing vectors in the ref frame
    vector<Vector3d>    _pts_cur;        // bearing vectors in the cur frame
    vector<Vector3d>    _xyz_in_cur;     // 3D points after homography
    vector<double>      _disparities;    // Disparities for each matching point
    vector<int>         _inliers;        // inlier indices
    SE3                 _T_cur_from_ref; // Transformation matrix (ref -> cur)
};

} // namespace init

} // namespace dr3

#endif // _INITIALIZATION_HPP_
