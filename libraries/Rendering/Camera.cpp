#include "Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui/imgui.h"

Camera::Camera(int width, int height) : m_width(width), m_height(height)
{
    m_sensitivity = 0.1f;

    Camera::reset();
}

void Camera::reset()
{
    m_pos = glm::vec3(0.0f);
    m_center = glm::vec3(0.0f);
    m_up = glm::vec3(0.0f, 1.0f, 0.0f);

    m_theta = glm::pi<float>() / 2.0f;
    m_phi = 0.f;
    m_viewMatrix = lookAt(m_pos, m_center, m_up);
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
            const float changeX = (static_cast<float>(x) - m_oldX) * 0.01f;
            const float changeY = (static_cast<float>(y) - m_oldY) * 0.01f;

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

glm::vec3 Camera::getPosition() const
{
    return m_pos;
}

glm::vec3 Camera::getCenter() const
{
    return m_center;
}

glm::vec3 Camera::getDirection() const
{
    return glm::normalize(m_center - m_pos);
}

void Camera::setPosition(glm::vec3 pos)
{
    m_pos = pos;
}

void Camera::setTheta(float theta)
{
	m_theta = theta;
}

void Camera::setPhi(float phi)
{
	m_phi = phi;
}

void Camera::setSensitivity(float sensitivity)
{
    m_sensitivity = sensitivity;
}

void Camera::setSensitivityFromBBox(glm::mat2x4 bbox)
{
    m_sensitivity = 0.001f * glm::length(bbox[1] - bbox[0]);
}
