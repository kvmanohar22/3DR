#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

namespace utils {

void frame_buffer_size_callback(GLFWwindow *window, int height, int width)
{
    glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int gl_renderer()
{
    // intialize glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "SLAM", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create window\n"
                  << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, frame_buffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -2;
    }

    while (!glfwWindowShouldClose(window))
    {
        process_input(window);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

class Frame
{
  private:
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;

  public:
    Frame(cv::Mat frame)
    {
        int MAX_CORNERS = 1000;
        cv::Mat des, gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(gray, corners, MAX_CORNERS, 0.01, 3);

        std::vector<cv::KeyPoint> kps;
        for (auto itr : corners)
        {
            kps.push_back(cv::KeyPoint(itr, 20));
        }

        std::vector<std::vector<cv::KeyPoint>> kpss;
        std::vector<cv::Mat> dess, imgs;
        imgs.push_back(gray);
        kpss.push_back(kps);

        cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
        orb->compute(imgs, kpss, dess);

        this->kps = kpss[0];
        this->des = dess[0];
    }

    std::vector<cv::KeyPoint> get_kps() { return this->kps; }
    cv::Mat get_des() { return this->des; }
};

}