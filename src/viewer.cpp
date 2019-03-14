#include "viewer.hpp"

namespace dr3 {

Viewer2D::Viewer2D() {
    cv::namedWindow("Viewer2D - SLAM");
}

void Viewer2D::update(cv::Mat img_l, cv::Mat img_r,
                      std::vector<cv::KeyPoint> kps_l,
                      std::vector<unsigned int> idxs_l,
                      std::vector<cv::KeyPoint> kps_r,
                      std::vector<unsigned int> idxs_r) {

   size_t new_h, new_w;
   new_h = img_l.rows + img_r.rows;
   new_w = img_l.cols;

   cv::Mat new_img(cv::Size(new_w, new_h), img_l.type());
   img_l.copyTo(new_img.rowRange(cv::Range(0, img_l.rows)));
   img_r.copyTo(new_img.rowRange(cv::Range(img_l.rows, img_l.rows+img_r.rows)));

   if (new_img.channels() < 3)
     cv::cvtColor(new_img, new_img, CV_GRAY2BGR);

   Viewer2D::draw_kps(new_img, kps_l, idxs_l, kps_r, idxs_r);
   cv::imshow("Viewer2D - SLAM", new_img);
   cv::waitKey(20);
}

void Viewer2D::draw_kps(cv::Mat &img, 
                        std::vector<cv::KeyPoint> kps_l, std::vector<unsigned int> idxs_l,
                        std::vector<cv::KeyPoint> kps_r, std::vector<unsigned int> idxs_r) {
   const int n_matches = idxs_l.size();
   for (int i = 0; i < n_matches; ++i) {
      cv::Point2f pt1 = kps_l[idxs_l[i]].pt;
      cv::Point2f pt2 = kps_r[idxs_r[i]].pt + cv::Point2f(cv::Size(0, img.rows/2));

      cv::circle(img, pt1, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(img, pt2, 2, cv::Scalar(255, 0, 0), -1);
      cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0));
   }
}

void Viewer2D::draw_point(cv::Mat &img, cv::KeyPoint pt) {
   cv::Point2f pt2f;
   pt2f.x = pt.pt.x;
   pt2f.y = pt.pt.y;
   draw_point(img, pt2f);
}

void Viewer2D::draw_point(cv::Mat &img, cv::Point2f pt) {
   cv::circle(img, pt, 2, cv::Scalar(255, 0, 0), -1);
}

Viewer3D::Viewer3D() {
   this->H = 720;
   this->W = 1024;
   Viewer3D::init();
}

Viewer3D::Viewer3D(size_t _H, size_t _W) : H(_H), W(_W) {
   Viewer3D::init();
}

void Viewer3D::init() {
   pangolin::CreateWindowAndBind("Viewer3D - SLAM", W, H);

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   s_cam = pangolin::OpenGlRenderState(
         pangolin::ProjectionMatrix(W, H, 420, 420, W/2, H/2, 0.1, 10000),
         pangolin::ModelViewLookAt(0, -10, -8,
                                   0,  0, 0,
                                   0, -1, 0));

   d_cam = pangolin::CreateDisplay();
   d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Pix(0), 1.0, (-1.0f * W) /H);
   d_cam.SetHandler(new pangolin::Handler3D(s_cam));

   // some constants
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Viewer3D::update() {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   d_cam.Activate(s_cam);
   pangolin::FinishFrame();
}

} // namespace dr3
