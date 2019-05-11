#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vikit/math_utils.h>
#include "camera.hpp"

namespace dr3 {

Pinhole::Pinhole(double width, double height,
                 double fx, double fy,
                 double cx, double cy,
                 double d0, double d1, double d2,
                 double d3, double d4) :
    AbstractCamera(width, height),
    _fx(fx), _fy(fy),
    _cx(cx), _cy(cy),
    _distortion(fabs(d0) > 1e-7) {
    _d[0] = d0; _d[1] = d1; _d[2] = d2;  _d[3] = d3; _d[4] = d4;
    _K << _fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0;
    _cvK = (cv::Mat_<float>(3, 3) << _fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0);
    _cvD = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);
}

Pinhole::~Pinhole() {}

Vector3d Pinhole::cam2world(const double &u, const double &v) const {
    Vector3d xyz;
    if (!_distortion) {
        xyz[0] = (u - _cx) / _fx;
        xyz[1] = (v - _cy) / _fy;
        xyz[2] = 1.0f;
    } else {
        cv::Point2f uv(u, v), px;
        const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
        cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
        cv::undistortPoints(src_pt, dst_pt, _cvK, _cvD);
        xyz[0] = px.x;
        xyz[1] = px.y;
        xyz[2] = 1.0;
    }
    return xyz.normalized();
}

Vector3d Pinhole::cam2world(const Vector2d &px) const {
    return cam2world(px[0], px[1]);
}

Vector2d Pinhole::world2cam(const Vector3d &xyz) const {
    return world2cam(vk::project2d(xyz));
}

Vector2d Pinhole::world2cam(const Vector2d &uv) const {
    Vector2d px;
    if (!_distortion) {
        px[0] = _fx * uv[0] + _cx;
        px[1] = _fy * uv[1] + _cy;
    } else {
        double x, y, r2, r4, r6, cdist, xd, yd, a1, a2, a3;
        x = uv[0];
        y = uv[1];
        r2 = x*x + y*y;
        r4 = r2*r2;
        r6 = r4*r2;
        a1 = 2*x*y;
        a2 = r2 + 2*x*x;
        a3 = r2 + 2*y*y;
        cdist = 1 + _d[0] * r2 + _d[1] * r4 + _d[4] * r6;
        xd = x * cdist + _d[2] * a1 + _d[3] * a2;
        yd = y * cdist + _d[2] * a3 + _d[3] * a1;
        px[0] = _fx * xd + _cx;
        px[1] = _fy * yd + _cy;
    }
    return px;
}


} // namespace dr3