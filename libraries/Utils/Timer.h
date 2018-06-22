#pragma once

#include <vector>
#include <glbinding/gl/gl.h>
using namespace gl;

#include "Utils/UtilCollection.h"

class Timer
{
public:
    Timer();
    ~Timer();

    /**
     * \brief starts the GPU timer
     */
    void start() const;

    /**
     * \brief stops the GPU timer
     */
    void stop();

    /**
     * \brief draws an imgui window with the frametime and a graph
     * \param window 
     */
    void drawGuiWindow(GLFWwindow* window);

	/**
	* \brief returns conent for imgui with the frametime and a graph
	* \param window
	* \param compact show compact gui in one line
	*/
	void drawGuiContent(GLFWwindow* window, bool compact = false);

private:
    std::vector<float> m_ftimes;
    GLuint m_query;
    GLuint m_elapsedTime = 0U;
    int m_done = false;
};
