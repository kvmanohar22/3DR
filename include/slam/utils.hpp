#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

namespace utils {

static inline bool comparator(cv::DMatch m1, cv::DMatch m2)
{
    return m1.distance < m2.distance;
}

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
        int MAX_CORNERS = 3000;
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

void match_frames(Frame f1, Frame f2, std::vector<int> &idx1, std::vector<int> &idx2) {

    std::vector<cv::DMatch> mmatches;
    cv::BFMatcher bf = cv::BFMatcher(cv::NORM_HAMMING, true);
    bf.match(f1.get_des(), f2.get_des(), mmatches);

    std::sort(mmatches.begin(), mmatches.end(), comparator);

    int top_matches = int((60 * mmatches.size()) / 100);
    std::vector<cv::DMatch> matches(mmatches.begin(), mmatches.begin() + top_matches);

    for (auto itr : matches) {
        idx1.push_back(itr.queryIdx);
        idx2.push_back(itr.trainIdx);
    }
}

cv::Mat estimate_fundamental_matrix(Frame f1, 
    Frame f2, const std::vector<int> idx1, 
    const std::vector<int> idx2, std::vector<uchar> &inliers) {

    std::vector<cv::KeyPoint> kps1 = f1.get_kps();
    std::vector<cv::KeyPoint> kps2 = f2.get_kps();

    std::vector<cv::Point2f> ps1, ps2;
    for (auto i : idx1) {
        float x = kps1[i].pt.x;
        float y = kps1[i].pt.y;
        ps1.push_back(cv::Point2f(x, y));
    }
    for (auto i : idx2) {
        float x = kps2[i].pt.x;
        float y = kps2[i].pt.y;
        ps2.push_back(cv::Point2f(x, y));
    }

    cv::Mat fmat = cv::findFundamentalMat(cv::Mat(ps1), cv::Mat(ps2), inliers);

    return fmat;
}

}
