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

cv::Mat Viewer2D::update(cv::Mat img_l, cv::Mat img_r,
                         std::vector<cv::Point2f> pts_l,
                         std::vector<cv::Point2f> pts_r) {

   size_t new_h, new_w;
   new_h = img_l.rows + img_r.rows;
   new_w = img_l.cols;

   cv::Mat new_img(cv::Size(new_w, new_h), img_l.type());
   img_l.copyTo(new_img.rowRange(cv::Range(0, img_l.rows)));
   img_r.copyTo(new_img.rowRange(cv::Range(img_l.rows, img_l.rows+img_r.rows)));

   cv::cvtColor(new_img, new_img, CV_GRAY2BGR);
   Viewer2D::draw_kps(new_img, pts_l, pts_r);
   update(new_img);
   return new_img.clone();
}

cv::Mat Viewer2D::update(cv::Mat img_l, cv::Mat img_r,
                         std::vector<cv::KeyPoint> kpts_l,
                         std::vector<cv::KeyPoint> kpts_r) {
    size_t new_h, new_w;
    new_h = img_l.rows + img_r.rows;
    new_w = img_l.cols;

    cv::Mat new_img(cv::Size(new_w, new_h), img_l.type());
    img_l.copyTo(new_img.rowRange(cv::Range(0, img_l.rows)));
    img_r.copyTo(new_img.rowRange(cv::Range(img_l.rows, img_l.rows+img_r.rows)));

    cv::cvtColor(new_img, new_img, CV_GRAY2BGR);
    Viewer2D::draw_kps(new_img, kpts_l, kpts_r);
    update(new_img);
    return new_img.clone();
}


void Viewer2D::update(cv::Mat &img) {
   cv::imshow("Viewer2D - SLAM", img);
   cv::waitKey(0);
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
                        std::vector<cv::Point2f> pts_l,
                        std::vector<cv::Point2f> pts_r) {
   const int n_matches = pts_l.size();
   for (int i = 0; i < n_matches; ++i) {
      cv::Point2f pt1 = pts_l[i];
      cv::Point2f pt2 = pts_r[i] + cv::Point2f(cv::Size(0, img.rows/2));

      cv::circle(img, pt1, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(img, pt2, 2, cv::Scalar(255, 0, 0), -1);
      cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0), 1, CV_AA);
   }
}

void Viewer2D::draw_kps(cv::Mat &img,
                        std::vector<cv::KeyPoint> kpts_l,
                        std::vector<cv::KeyPoint> kpts_r) {
   const int n_matches = kpts_l.size();
   for (int i = 0; i < n_matches; ++i) {
      cv::Point2f pt1 = kpts_l[i].pt;
      cv::Point2f pt2 = kpts_r[i].pt + cv::Point2f(cv::Size(0, img.rows/2));

      cv::circle(img, pt1, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(img, pt2, 2, cv::Scalar(255, 0, 0), -1);
      cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0), 1, CV_AA);
   }
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
      cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0), 1, CV_AA);
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


Viewer3D::Viewer3D(Map *mapp) {
   this->mapp = mapp;
   Viewer3D::setup();
}

void Viewer3D::setup() {
   window_name = "SLAM: 3D";
   H = 768;
   W = 1024;

   pangolin::CreateWindowAndBind(window_name, W, H);
   glEnable(GL_DEPTH_TEST);
   pangolin::GetBoundWindow()->RemoveCurrent();
}

void Viewer3D::draw_camera(cv::Mat Rt, cv::Point3f color) {
   float w = 1.0;
   float h = w * 0.75;
   float z = w * 0.6;

   Rt = Rt.t();

   glPushMatrix();

   glMultMatrixf(Rt.ptr<GLfloat>(0));

   glLineWidth(1.0f);
   glColor3f(color.x, color.y, color.z);
   glBegin(GL_LINES);
   glVertex3f(0,0,0);
   glVertex3f(w,h,z);
   glVertex3f(0,0,0);
   glVertex3f(w,-h,z);
   glVertex3f(0,0,0);
   glVertex3f(-w,-h,z);
   glVertex3f(0,0,0);
   glVertex3f(-w,h,z);

   glVertex3f(w,h,z);
   glVertex3f(w,-h,z);

   glVertex3f(-w,h,z);
   glVertex3f(-w,-h,z);

   glVertex3f(-w,h,z);
   glVertex3f(w,h,z);

   glVertex3f(-w,-h,z);
   glVertex3f(w,-h,z);
   glEnd();

   glPopMatrix();
}

void Viewer3D::update() {
   pangolin::BindToContext(window_name);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   s_cam = pangolin::OpenGlRenderState(
       pangolin::ProjectionMatrix(W, H, 420, 420, 512, 389, 0.1, 1000),
       pangolin::ModelViewLookAt(0, -1, -2, // camera location in world
                                 0,  0,  0, // where should the camera look at?
                                 0, -1,  0) // camera y-axis
       );

   pangolin::Handler3D handler(s_cam);
   d_cam = pangolin::CreateDisplay()
       .SetBounds(0.0, 1.0, 0.0, 1.0, -W/H)
       .SetHandler(&handler);

   cv::Point3f red, green, blue;
   red.x   = 1.0f; red.y   = 0.0f; red.z   = 0.0f;
   green.x = 0.0f; green.y = 1.0f; green.z = 0.0f;
   blue.x  = 0.0f; blue.y  = 0.0f; blue.z  = 1.0f;
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

   while(!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      d_cam.Activate(s_cam);

      // Draw world axes
      draw_axes();

      // Render cameras
      const std::vector<FramePtr> frames = mapp->get_frames();
      for (auto &itr : frames) {
         cv::Mat pose = itr->get_pose_c2w();
            draw_camera(pose, green);
      }
      glEnd();

      // Render points
      glPointSize(2.0f);
      glBegin(GL_POINTS);
      glColor3f(1.0f, 1.0f, 1.0f);
      const std::vector<Point*> &points = mapp->get_points();
      for (auto &itr : points) {
         if (!itr->is_valid())
            continue;

         cv::Mat point = itr->get_xyz();
         glVertex3f(point.at<float>(0),
                    point.at<float>(1),
                    point.at<float>(2));
      }
      glEnd();
      pangolin::FinishFrame();
   }
}

void Viewer3D::draw_axes() {
      glBegin(GL_LINES); 

      // x-axis
      glColor3f(1.0f, 0.0f, 0.0f);
      glVertex3f(0, 0, 0);
      glVertex3f(1, 0, 0);

      // y-axis
      glColor3f(0.0f, 1.0f, 0.0f);
      glVertex3f(0, 0, 0);
      glVertex3f(0, 1, 0);

      // z-axis
      glColor3f(0.0f, 0.0f, 1.0f);
      glVertex3f(0, 0, 0);
      glVertex3f(0, 0, 1);

      glEnd();
}



void Viewer3D::check_axes() {
   pangolin::BindToContext(window_name);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   s_cam = pangolin::OpenGlRenderState(
       pangolin::ProjectionMatrix(W, H, 420, 420, 512, 389, 0.1, 1000),
       pangolin::ModelViewLookAt(0, -1, -2, // camera location in world
                                 0,  0,  0, // where should the camera look at?
                                 0, -1,  0) // camera y-axis
       );

   pangolin::Handler3D handler(s_cam);
   d_cam = pangolin::CreateDisplay()
       .SetBounds(0.0, 1.0, 0.0, 1.0, -W/H)
       .SetHandler(&handler);

   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

   while(!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      d_cam.Activate(s_cam);
      glBegin(GL_LINES);
      
      draw_axes();

      pangolin::FinishFrame();
   }
}

} // namespace dr3
