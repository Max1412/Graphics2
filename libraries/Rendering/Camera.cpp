#include "Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui/imgui.h"

Camera::Camera(int width, int height, float radius)
{
    m_pos = glm::vec3(0.0f, 0.0f, radius);
    m_center = glm::vec3(0.0f, 0.0f, 0.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);

    m_sensitivity = 0.1f;
    m_theta = glm::pi<float>() / 2.0f;
    m_phi = 0.f;
    m_radius = radius;
    m_width = width;
    m_height = height;

    m_viewMatrix = lookAt(m_center + m_pos, m_center, m_up);
    m_oldX = width / 2.f;
    m_oldY = height / 2.f;
}

void Camera::reset()
{
    m_pos = glm::vec3(0.0f, 0.0f, m_radius);
    m_center = glm::vec3(0.0f, 0.0f, 0.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);

    m_theta = glm::pi<float>() / 2.0f;
    m_phi = 0.f;
    m_viewMatrix = lookAt(m_center + m_pos, m_center, m_up);
    m_oldX = m_width / 2.f;
    m_oldY = m_height / 2.f;
}

void Camera::update(GLFWwindow* window)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            float changeX = (static_cast<float>(x) - m_oldX) * m_sensitivity * 0.1f;
            float changeY = (static_cast<float>(y) - m_oldY) * m_sensitivity * 0.1f;

            m_theta -= changeY;
            if (m_theta < 0.01f)
            {
                m_theta = 0.01f;
            }
            else if (m_theta > glm::pi<float>() - 0.01f)
            {
                m_theta = glm::pi<float>() - 0.01f;
            }

            m_phi -= changeX;
            if (m_phi < 0)
            {
                m_phi += 2 * glm::pi<float>();
            }
            else if (m_phi > 2 * glm::pi<float>())
            {
                m_phi -= 2 * glm::pi<float>();
            }
        }

        m_oldX = static_cast<float>(x);
        m_oldY = static_cast<float>(y);

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

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            m_center += glm::normalize(m_center - m_pos) * m_sensitivity;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            m_center -= glm::normalize(m_center - m_pos) * m_sensitivity;
        }

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            m_center += glm::normalize(glm::cross(m_up, m_center - m_pos)) * m_sensitivity;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            m_center -= glm::normalize(glm::cross(m_up, m_center - m_pos)) * m_sensitivity;
        }

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            m_center += glm::normalize(m_up) * m_sensitivity;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        {
            m_center -= glm::normalize(m_up) * m_sensitivity;
        }

        m_pos.x = m_center.x + m_radius * sin(m_theta) * sin(m_phi);
        m_pos.y = m_center.y + m_radius * cos(m_theta);
        m_pos.z = m_center.z + m_radius * sin(m_theta) * cos(m_phi);

        m_viewMatrix = lookAt(m_pos, m_center, m_up);
    }
}

const glm::mat4& Camera::getView() const
{
    return m_viewMatrix;
}

glm::mat4& Camera::getView()
{
    return m_viewMatrix;
}

glm::vec3& Camera::getPosition()
{
    return m_pos;
}
