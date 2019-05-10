#ifndef _CAMERAS_HPP_
#define _CAMERAS_HPP_

#include <Eigen/Core>
#include <iostream>

namespace dr3 {

using namespace std;
using namespace Eigen;

/*
    - Base camera
    - All cameras should derive from this
*/
class AbstractCamera {
protected:
    int _width;
    int _height;

public:
    AbstractCamera() : _width(0), _height(0) {}
    AbstractCamera(int width, int height) :
        _width(width), _height(height) {}

    virtual ~AbstractCamera() =default;

    virtual Vector3d cam2world(const double &u, const double &v) const =0;
    virtual Vector3d cam2world(const Vector2d &px) const =0;
    virtual Vector2d world2cam(const Vector3d &xyz) const =0;

    virtual double error2() const =0;

    inline int width()  { return _width;  }
    inline int height() { return _height; }

    inline bool is_in_frame(const Vector2i &obs, int boundary=0) const {
        if (obs[0] >= boundary && obs[0] < _width - boundary &&
            obs[1] >= boundary && obs[1] < _height - boundary)
            return true;
        return false;
    }

    inline bool is_in_frame(const Vector2i &obs, int boundary, int level) const {
        if (obs[0] >= boundary && obs[0] < _width / (1 << level) - boundary &&
            obs[1] >= boundary && obs[1] < _height / (1 << level) - boundary)
            return true;
        return false;
    }
};

/*
    Pinhole camera model

    TODO: Distortion not handled
*/
class Pinhole : public AbstractCamera {
private:
    const double _fx, _fy;
    const double _cx, _cy;
    bool _distortion;
    Matrix3d _K;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pinhole(double width, double height,
            double fx, double fy, double cx, double cy,
            double d0=0.0f, double d1=0.0f, double d2=0.0f, 
            double d3=0.0f, double d4=0.0f);
    ~Pinhole() override;

    Vector3d cam2world(const double &u, const double &v) const override;
    Vector3d cam2world(const Vector2d &px) const override;
    Vector2d world2cam(const Vector3d &xyz) const override;

    double error2() const override { return fabs(_fx); }

    inline double fx() const { return _fx; }
    inline double fy() const { return _fy; }
    inline double cx() const { return _cx; }
    inline double cy() const { return _cy; }
};

} // namespace dr3

#endif