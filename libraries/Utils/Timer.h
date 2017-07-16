#pragma once

#include <vector>
#include <iostream>
#include <GL/glew.h>

#include "Utils/UtilCollection.h"

#include "imgui/imgui_impl_glfw_gl3.h"



class Timer {
public:
    Timer();

    void del();
    void start();
    void stop();
    void drawGuiWindow(GLFWwindow* window);

private:
    std::vector<float> m_ftimes;
    GLuint m_query;
    GLuint m_elapsedTime;
    int m_done = false;
};