#include <config.hpp>

namespace dr3 {

Config *Config::_instance = NULL;

Config::Config() :
    __ransac_iters(50),
    __ransac_threshold(5.0f),
    __cell_size(30),
    __n_pyr_levels(3),
    __min_harris_corner_score(20.0),
    __reprojection_threshold(0.2)
{}

Config* Config::get_instance() {
    if (_instance == NULL)
        _instance = new Config(); // Generate only one instance of the class
    return _instance;
}

} // namespace dr3
