#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class SimpleTrackball
{
public:
    SimpleTrackball(int width, int height, float radius);

    /**
     * \brief Updates the view matrix based on mouse input
     * \param window 
     */
    void update(GLFWwindow* window);

    /**
     * \brief returns the view matrix (const)
     * \return 4x4 view matrix (const)
     */
    const glm::mat4& getView() const;

    /**
     * \brief returns the view matrix (mutable)
     * \return 4x4 view matrix
     */
    glm::mat4& getView();

    glm::vec3& getPosition();

private:
    glm::mat4 m_viewMatrix;
    glm::vec3 m_center;
    glm::vec3 m_eye;
    glm::vec3 m_pos;
    glm::vec3 m_up;

    float m_sensitivity;
    float m_theta;
    float m_phi;
    float m_radius;
    float m_oldX;
    float m_oldY;
};
