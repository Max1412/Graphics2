#pragma once
#include "Camera.h"

class Pilotview : public Camera
{
public:
    Pilotview(int width, int height);

    /**
     * \brief Updates the view matrix based on mouse input
     * \param window 
     */
    void update(GLFWwindow* window) override;

    void reset() override;

    void setDirection(glm::vec3 dir);

private:
    glm::vec3 m_dir;
};
