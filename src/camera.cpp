#include "camera.hpp"

namespace dr3 {

Pinhole::Pinhole(double width, double height,
                 double fx, double fy,
                 double cx, double cy,
                 double d0, double d1, double d2,
                 double d3, double d4) :
    AbstractCamera(width, height),
    _fx(fx), _fy(fy),
    _cx(cx), _cy(cy) {

    _d[0] = d0;
    _d[1] = d1;
    _d[2] = d2;
    _d[3] = d3;
    _d[4] = d4;

    _K << _fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0;
}

Pinhole::~Pinhole() {}

Vector3d Pinhole::cam2world(const double &u, const double &v) const {
    Vector3d xyz;
    if (!_distortion) {
        xyz[0] = (u - _cx) / _fx;
        xyz[1] = (v - _cy) / _fy;
        xyz[2] = 1.0f;
    } else {
        std::cerr << "Not handled" << std::endl;
    }
    return xyz.normalized();
}

Vector3d Pinhole::cam2world(const Vector2d &px) const {
    return cam2world(px[0], px[1]);
}

Vector2d Pinhole::world2cam(const Vector3d &xyz) const {
    // TODO
}


} // namespace dr3