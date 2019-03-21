#include "viewer.hpp"
#include "utils.hpp"

#include <thread>

using namespace std;
using namespace dr3;

string window_name = "pangolin";

void setup() {
   // create a window and bind its context to the main thread
   pangolin::CreateWindowAndBind(window_name, 640, 480);

   // enable depth
   glEnable(GL_DEPTH_TEST);

   // unset the current context from the main thread
   pangolin::GetBoundWindow()->RemoveCurrent();
}

void run() {
   // fetch the context and bind it to this thread
   pangolin::BindToContext(window_name);

   // we manually need to restore the properties of the context
   glEnable(GL_DEPTH_TEST);

   // Define Projection and initial ModelView matrix
   pangolin::OpenGlRenderState s_cam(
     pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
     pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
   );

   // Create Interactive View in window
   pangolin::Handler3D handler(s_cam);
   pangolin::View& d_cam = pangolin::CreateDisplay()
         .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
         .SetHandler(&handler);

   while( !pangolin::ShouldQuit() )
   {
     // Clear screen and activate view to render into
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     d_cam.Activate(s_cam);

     // Render OpenGL Cube
     pangolin::glDrawColouredCube();

     // Swap frames and Process Events
     pangolin::FinishFrame();
   }

   // unset the current context from the main thread
   pangolin::GetBoundWindow()->RemoveCurrent();   
}

int main() {

   // Viewer2D v2d;
   // cv::Mat img = utils::load_image("../imgs/slam/img_l.png");

   // // draw point
   // cv::Point2f pt2d;
   // pt2d.x = 323;
   // pt2d.y = 133;
   // v2d.draw_point(img, pt2d);
   // utils::view_image("img", img);

   // cv::Point3f pt3d;
   // pt3d.x = 1;
   // pt3d.y = 1;
   // pt3d.z = 500;
   // v2d.draw_line(img, pt3d);
   // utils::view_image("img", img);

   // Viewer3D v3d;
   // // v3d.update();
   // std::thread render_loop;
   // render_loop = std::thread(&Viewer3D::update, v3d);
   // render_loop.join();


    // create window and context in the main thread
    // setup();

    // use the context in a separate rendering thread
    // std::thread render_loop;
    // render_loop = std::thread(run);
    // render_loop.join();

    // return 0;
}
