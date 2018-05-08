#pragma once

#include "Camera.h"

class Trackball : public Camera
{
public:
    Trackball(int width, int height, float radius);

    /**
     * \brief Updates the view matrix based on mouse input
     * \param window 
     */
    void update(GLFWwindow* window) override;

    void reset() override;

    void setCenter(glm::vec3 center);

private:

    float m_radius;
};
