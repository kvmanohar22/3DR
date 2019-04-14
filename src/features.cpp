#include "features.hpp"

namespace dr3 {

namespace feature_detection {

Detector::Detector(
    const int img_w,
    const int img_h,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size(cell_size),
        n_pyr_levels(n_pyr_levels),
        grid_n_cols(ceil(static_cast<double>(img_w) / cell_size)),
        grid_n_rows(ceil(static_cast<double>(img_h) / cell_size)),
        grid_occupancy(grid_n_cols * grid_n_rows, false)
    {}

void Detector::reset_grid() {
    std::fill(grid_occupancy.begin(), grid_occupancy.end(), false);
}

void Detector::flag_grid(const Vector2d &px) {
    grid_occupancy.at(
        static_cast<int>(px[1] / cell_size) * grid_n_cols
      + static_cast<int>(px[0] / cell_size)) = true;
}

void Detector::flag_features_grid(const Features &fts) {
    std::for_each(fts.begin(), fts.end(), [&](Feature *ft) {
        flag_grid(ft->px);
    });
}

FastDetector::FastDetector(
    const int img_w,
    const int img_h,
    const int cell_size,
    const int n_pyr_levels) :
        Detector(img_w, img_h, cell_size, n_pyr_levels)
    {}

void FastDetector::detect(
    FramePtr frame,
    const ImgPyramid &img_pyr,
    const double detection_threshold,
    Features &fts) {

    Corners corners(grid_n_cols * grid_n_rows, Corner(0, 0, detection_threshold, 0));

    for (int lvl = 0; lvl < n_pyr_levels; ++lvl) {
        const int scale = (1 << lvl);
        std::vector<fast::fast_xy> fast_corners;

        #if __SSE2__
            fast::fast_corner_detect_10_sse2(
                (fast::fast_byte*) img_pyr[lvl].data, img_pyr[lvl].cols,
                img_pyr[lvl].rows, img_pyr[lvl].cols, 20, fast_corners);
        #elif HAVE_FAST_NEON
            fast::fast_corner_detect_9_neon(
                (fast::fast_byte*) img_pyr[lvl].data, img_pyr[lvl].cols,
                img_pyr[lvl].rows, img_pyr[lvl].cols, 20, fast_corners);
        #else
            fast::fast_corner_detect_10(
                (fast::fast_byte*) img_pyr[lvl].data, img_pyr[lvl].cols,
                img_pyr[lvl].rows, img_pyr[lvl].cols, 20, fast_corners);
        #endif

        std::vector<int> scores, num_corners;
        fast::fast_corner_score_10((fast::fast_byte*)img_pyr[lvl].data,
                                   img_pyr[lvl].cols,
                                   fast_corners, 20, scores);
        fast::fast_nonmax_3x3(fast_corners, scores, num_corners);

        for (auto &itr: num_corners) {
            fast::fast_xy &xy = fast_corners.at(itr);
            const int k = static_cast<int>((xy.y * scale) / cell_size) * grid_n_cols
                        + static_cast<int>((xy.x * scale) / cell_size);
            if (grid_occupancy[k])
                continue;
            const float score = utils::shi_tomasi_score(img_pyr[lvl], xy.x, xy.y);
            if (score > corners.at(k).score)
                corners.at(k) = Corner(xy.x * scale, xy.y * scale, score, lvl);
        }
    }

    // Create feature for each corner
    std::for_each(corners.begin(), corners.end(), [&](Corner &corner) {
        if (corner.score > detection_threshold) {
            fts.push_back(new Feature(frame,
                                      Vector2d(corner.x, corner.y),
                                      corner.level)
            );
        }
    });

    reset_grid();
}

} // namespace feature_detection

} // namespace dr3
