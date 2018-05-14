#include "Trackball.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../Utils/UtilCollection.h"

Trackball::Trackball(int width, int height, float radius) : Camera(width, height), m_radius(radius)
{
    Trackball::reset();
}

void Trackball::reset()
{
    Camera::reset();
    m_pos = glm::vec3(0.0f, 0.0f, m_radius);
    m_viewMatrix = lookAt(m_center + m_pos, m_center, m_up);
}

void Trackball::setCenter(glm::vec3 center)
{
    m_center = center;
}

void Trackball::update(GLFWwindow* window)
{
    Camera::update(window);

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        m_radius -= m_sensitivity;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        m_radius += m_sensitivity;
    }
    if (m_radius < 0.1f)
    {
        m_radius = 0.1f;
    }

    m_pos.x = m_center.x + m_radius * sin(m_theta) * sin(m_phi);
    m_pos.y = m_center.y + m_radius * cos(m_theta);
    m_pos.z = m_center.z + m_radius * sin(m_theta) * cos(m_phi);

    m_viewMatrix = lookAt(m_pos, m_center, m_up);

    if constexpr(util::debugmode)
    {
        // TODO implement proper "pilot view" Trackball so the trackball doesn't overflow
        if (std::isnan(m_viewMatrix[0][0]))
            throw std::runtime_error("NaN in View Matrix");
    }
}
