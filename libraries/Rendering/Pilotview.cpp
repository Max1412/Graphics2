#include "Pilotview.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../Utils/UtilCollection.h"

Pilotview::Pilotview(int width, int height) : Camera(width, height)
{
    Pilotview::reset();
}

void Pilotview::reset()
{
    Camera::reset();
    m_dir = glm::vec3(0.0f, 0.0f, -1.0f);
    m_pos = glm::vec3(0.0f);
    m_center = m_pos + m_dir;
    m_viewMatrix = lookAt(m_pos, m_center, m_up);
    m_phi = glm::pi<float>();
}

void Pilotview::setDirection(glm::vec3 dir)
{
    m_dir = dir;
}

void Pilotview::update(GLFWwindow* window)
{
    Camera::update(window);

    m_dir.x = sin(m_theta) * sin(m_phi);
    m_dir.y = -cos(m_theta);
    m_dir.z = sin(m_theta) * cos(m_phi);
    m_dir = glm::normalize(m_dir);

    float old_sensitivity = m_sensitivity;

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        m_sensitivity *= 100; // fast mode
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        m_sensitivity *= 0.1f; // slow mode
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        m_pos += m_dir * m_sensitivity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        m_pos -= m_dir * m_sensitivity;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        m_pos += glm::normalize(glm::cross(m_up, m_dir)) * m_sensitivity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        m_pos -= glm::normalize(glm::cross(m_up, m_dir)) * m_sensitivity;
    }

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        m_pos += glm::normalize(m_up) * m_sensitivity;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        m_pos -= glm::normalize(m_up) * m_sensitivity;
    }

    m_sensitivity = old_sensitivity;

    m_center = m_pos + m_dir;
    m_viewMatrix = lookAt(m_pos, m_center, m_up);

    if constexpr(util::debugmode)
    {
        // TODO implement proper "pilot view" Pilotview so the trackball doesn't overflow
        if (std::isnan(m_viewMatrix[0][0]))
            throw std::runtime_error("NaN in View Matrix");
    }
}
