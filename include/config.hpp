#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <string>
#include <stdint.h>
#include <stdio.h>

namespace dr3 {

using std::string;

class Config {
public:
    static Config* get_instance();

    static int& ransac_iters() { return get_instance()->__ransac_iters; }
    static float& ransac_threshold() { return get_instance()->__ransac_threshold; }
    static int& cell_size() { return get_instance()->__cell_size; }
    static int& n_pyr_levels() { return get_instance()->__n_pyr_levels; }
    static double& min_harris_corner_score() { return get_instance()->__min_harris_corner_score; }
    static double& reprojection_threshold() { return get_instance()->__reprojection_threshold; }
    static double& map_scale() { return get_instance()->__map_scale; }

private:
    Config();
    Config(Config const&);
    Config& operator = (Config const&);
    static Config *_instance;

    int    __ransac_iters;
    float  __ransac_threshold;
    int    __cell_size;
    int    __n_pyr_levels;
    double __min_harris_corner_score;
    double __reprojection_threshold;   // in pixels
    double __map_scale;                // Scene depth for initial map estimation
};

} // namespace dr3

#endif
