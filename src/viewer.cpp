#include "viewer.hpp"

namespace dr3 {

Viewer2D::Viewer2D() {}

cv::Mat Viewer2D::update(cv::Mat img_l, cv::Mat img_r,
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

   Viewer2D::draw_kps(new_img, kps_l, idxs_l, kps_r, idxs_r);
   update(new_img);
   return new_img.clone();
}

void Viewer2D::update(cv::Mat &img) {
   cv::imshow("Viewer2D - SLAM djflkjd", img);
   cv::waitKey(20);
}

void Viewer2D::update(cv::Mat &img,
                      const std::vector<cv::KeyPoint> &kps) {
   draw_kps(img, kps);
   update(img);
}

void Viewer2D::draw_kps(cv::Mat &img, 
                        const std::vector<cv::KeyPoint> &kps) {
   if (img.channels() < 3)
     cv::cvtColor(img, img, CV_GRAY2BGR);

   for (int i = 0; i < kps.size(); ++i)
      draw_point(img, kps[i].pt);
}


void Viewer2D::draw_kps(cv::Mat &img, 
                        std::vector<cv::KeyPoint> kps_l,
                        std::vector<unsigned int> idxs_l,
                        std::vector<cv::KeyPoint> kps_r,
                        std::vector<unsigned int> idxs_r) {
   const int n_matches = idxs_l.size();
   for (int i = 0; i < n_matches; ++i) {
      cv::Point2f pt1 = kps_l[idxs_l[i]].pt;
      cv::Point2f pt2 = kps_r[idxs_r[i]].pt + cv::Point2f(cv::Size(0, img.rows/2));

      cv::circle(img, pt1, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(img, pt2, 2, cv::Scalar(255, 0, 0), -1);
      cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0));
   }
}

void Viewer2D::draw_point(cv::Mat &img,
                          cv::KeyPoint pt,
                          cv::Scalar color) {
   cv::Point2f pt2f;
   pt2f.x = pt.pt.x;
   pt2f.y = pt.pt.y;
   draw_point(img, pt2f, color);
}

void Viewer2D::draw_point(cv::Mat &img,
                          cv::Point2f pt,
                          cv::Scalar color) {
   cv::circle(img, pt, 2, color, -1);
}

void Viewer2D::draw_line(cv::Mat &img,
                         cv::Point3f line,
                         cv::Scalar color) {
   float a = line.x;
   float b = line.y;
   float c = line.z;

   cv::Point2f pt1, pt2;
   pt1.x = 0;
   pt1.y = int(-c / b);
   pt2.x = img.cols;
   pt2.y = int(-(c + a * img.cols) / b);
   cv::line(img, pt1, pt2, color, 1, CV_AA);
}

Viewer3D::Viewer3D() {
   this->H = 768;
   this->W = 1024;
   Viewer3D::init();
}

Viewer3D::Viewer3D(float _H, float _W)
  : H(_H), W(_W) {
   Viewer3D::init();
}

void Viewer3D::init() {
   pangolin::CreateWindowAndBind("Viewer3D - SLAM", W, H);

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   s_cam = pangolin::OpenGlRenderState(
         pangolin::ProjectionMatrix(W, H, 420, 420, 512, 389, 0.1, 100),
         pangolin::ModelViewLookAt(4,  0, 0,
                                   0,  0, 0,
                                   0, -1, 0));

   d_cam = pangolin::CreateDisplay();
   // d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0, (-1.0f * W) /H);
   d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0, -W/H);
   // d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, (1.0f * W) /H);
   d_cam.SetHandler(new pangolin::Handler3D(s_cam));
}

void Viewer3D::update() {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   d_cam.Activate(s_cam);
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

   glPointSize(50);
   glBegin(GL_POINTS);
   glColor3f(0.0, 1.0, 0.0);
   glVertex3f(4.0, 0.0, 0.0);
   glEnd();

   pangolin::FinishFrame();
}

void Viewer3D::update(cv::Mat &Rt, std::vector<cv::Mat> &pts3d) {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   d_cam.Activate(s_cam);

   // render points
   glBegin(GL_POINTS);
   glPointSize(3.0f);
   glColor3f(1.0, 0.0, 1.0);
   for (int idx = 0; idx < pts3d.size(); ++idx) {
      const int n_points = pts3d[idx].rows;
      for (int i = 0; i < n_points; ++i) {
         float x = pts3d[idx].at<float>(i, 0);
         float y = pts3d[idx].at<float>(i, 1);
         float z = pts3d[idx].at<float>(i, 2);
         glVertex3f(x, y, z);
      }
   }
   glEnd();

   // render camera


   pangolin::FinishFrame();
}

} // namespace dr3
