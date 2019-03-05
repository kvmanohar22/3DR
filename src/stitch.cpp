#include "stitch.hpp"

namespace reconstruct {

int Stitch::process(cv::Mat _left_img, cv::Mat _right_img) {

    cv::Mat left_gray, right_gray;
    cv::cvtColor(_left_img, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(_right_img, right_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
    cv::Mat left_descriptors, right_descriptors;
    orb->detectAndCompute(left_gray, cv::noArray(), left_keypoints, left_descriptors);
    orb->detectAndCompute(right_gray, cv::noArray(), right_keypoints, right_descriptors);

    std::vector<cv::DMatch> mmatches;
    cv::BFMatcher bf = cv::BFMatcher(cv::NORM_HAMMING, true);
    bf.match(left_descriptors, right_descriptors, mmatches);

    std::sort(mmatches.begin(), mmatches.end(), Stitch::comparator);

    // Note: To reduce the noise in the matches, we consider only top
    //       few percent of matches.
    int top_matches = int((20 * mmatches.size()) / 100);
    std::vector<cv::DMatch> matches(mmatches.begin(), mmatches.begin() + top_matches);

    cv::Mat H = Stitch::align_pair(left_keypoints, right_keypoints, matches);
    cv::Size right_size = right_gray.size();
    float h = right_size.height, w = right_size.width;

    // TODO: Handle the case when the matrix is singular
    cv::Mat Hinv = H.inv();
    Hinv = Hinv * (1.0 / Hinv.at<float>(2, 2));

    this->H = Hinv;

    float points[][3] = {0, 0, 1, w, 0, 1, 0, h, 1, w, h, 1};
    cv::Mat points_mat(cv::Size(3, 4), CV_32F, &points);

    cv::Mat tr_points = (Hinv * points_mat.t()).t();

    for (int i = 0; i < tr_points.rows; ++i) {
        float weight = tr_points.at<float>(i, 2);
        for (int j = 0; j < tr_points.cols - 1; ++j) {
            tr_points.at<float>(i, j) /= weight;
        }
    }

    std::vector<float> X, Y;
    for (int i = 0; i < tr_points.rows; ++i) {
        X.push_back(tr_points.at<float>(i, 0));
        Y.push_back(tr_points.at<float>(i, 1));
    }
    for (int i = 0; i < points_mat.rows; ++i) {
        X.push_back(points_mat.at<float>(i, 0));
        Y.push_back(points_mat.at<float>(i, 1));
    }

    auto min_X = *std::min_element(std::begin(X), std::end(X));
    auto min_Y = *std::min_element(std::begin(Y), std::end(Y));
    auto max_X = *std::max_element(std::begin(X), std::end(X));
    auto max_Y = *std::max_element(std::begin(Y), std::end(Y));

    int new_width = int(std::ceil(max_X) - std::floor(min_X));
    int new_height = int(std::ceil(max_Y) - std::floor(min_Y));

    float translate_top_left[][3] = {1, 0, -min_X, 0, 1, -min_Y, 0, 0, 1};
    cv::Mat translate_top_left_mat(cv::Size(3, 3), CV_32F, &translate_top_left);

    cv::Mat warped_right_image, warped_left_image;
    cv::warpPerspective(_right_img, warped_right_image, translate_top_left_mat * Hinv, {new_width, new_height});
    cv::warpPerspective(_left_img, warped_left_image, translate_top_left_mat, {new_width, new_height});

    float alpha = 0.5f;
    float beta = 1.0f - alpha;
    float gamma = 0.0f;

    cv::addWeighted(warped_left_image, alpha, warped_right_image, beta, gamma, this->final_stitched_img);

    return 0;
}

int Stitch::process(std::string left_image_file,
    std::string right_image_file) {

    if (!is_valid()) {
        std::cerr << "Invalid camera parameters" << std::endl;
        return -1;
    }

    // Load images
    this->load_left_image(left_image_file);
    this->load_right_image(right_image_file);

    return this->process(this->_left_img, this->_right_img);
}


cv::Mat Stitch::align_pair(std::vector<cv::KeyPoint> left_keypoints,
    std::vector<cv::KeyPoint> right_keypoints,
    std::vector<cv::DMatch> matches) {

    size_t max_inliers = 0;
    std::vector<int> best_inliers;
    int max_index = matches.size();

    for (int i = 0; i < ransac_iters; ++i) {
        // Generate random subset of matches
        std::set<int> indexes;
        std::vector<cv::DMatch> match_subset_sample;
        while (indexes.size() < std::min(min_matches, max_index)) {
            int random_index = rand() % max_index;
            if (indexes.find(random_index) == indexes.end()) {
                match_subset_sample.push_back(matches[random_index]);
                indexes.insert(random_index);
            }
        }

        cv::Mat H_estimate = cv::Mat::eye(cv::Size(3, 3), CV_32F);
        H_estimate = utils::compute_homography(left_keypoints, 
            right_keypoints, match_subset_sample);

        std::vector<int> inliers = get_inliers(left_keypoints,
            right_keypoints, matches, H_estimate);

        size_t n_inliers = inliers.size();
        if (n_inliers > max_inliers) {
            max_inliers = n_inliers;
            best_inliers = inliers;
        }
    }

    cv::Mat H_best = least_squares_fit(left_keypoints,
        right_keypoints, matches, best_inliers);

    return H_best;
}


std::vector<int> Stitch::get_inliers(std::vector<cv::KeyPoint> f1,
    std::vector<cv::KeyPoint> f2,
    std::vector<cv::DMatch> matches,
    cv::Mat M) {

    std::vector<int> inlier_indices;
    for (int i = 0; i < matches.size(); ++i) {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;

        float query_point[] = {f1[query_idx].pt.x,
                               f1[query_idx].pt.y,
                               1};
        cv::Mat query_mat(cv::Size(1, 3), CV_32F, &query_point);
        cv::Mat transformed_mat = M * query_mat;

        const float *ptr = transformed_mat.ptr<float>(0);
        float query_point_transform[] = {ptr[0]/ptr[2], ptr[1]/ptr[2]};
        float original_point[] = {f2[train_idx].pt.x, f2[train_idx].pt.y};
        cv::Mat pt1(cv::Size(1, 2), CV_32F, &query_point_transform);
        cv::Mat pt2(cv::Size(1, 2), CV_32F, &original_point);

        if (cv::norm(pt1-pt2) < ransac_thresh) {
            inlier_indices.push_back(i);
        }
    }

    return inlier_indices;
}


cv::Mat Stitch::least_squares_fit(std::vector<cv::KeyPoint> f1,
    std::vector<cv::KeyPoint> f2,
    std::vector<cv::DMatch> matches,
    std::vector<int> best_inliers) {

    std::vector<cv::DMatch> inlier_subset_samples;
    for (std::vector<int>::const_iterator itr = best_inliers.begin();
            itr != best_inliers.end(); ++itr)
        inlier_subset_samples.push_back(matches[*itr]);

    cv::Mat H_best = utils::compute_homography(f1, f2, inlier_subset_samples);

    return H_best;
}

} // namespace reconstruct
