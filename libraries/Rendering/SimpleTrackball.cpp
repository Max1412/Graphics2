#include "SimpleTrackball.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "imgui/imgui.h"


SimpleTrackball::SimpleTrackball(int width, int height, float radius) {

	m_pos = glm::vec3(0.0f, 0.0f, 5.0);
	m_center = glm::vec3(0.0f, 0.0f, 0.0f);
	m_up = glm::vec3(0.0f, 1.0f, 0.0f);

	m_sensitivity = 0.010f;
	m_theta = glm::pi<float>() / 2.0f;
	m_phi = 0.f;
	m_radius = radius;

	m_viewMatrix = glm::lookAt(m_center + m_pos, m_center, m_up);

	m_oldX = width / 2.f;
	m_oldY = height / 2.f;
}

void SimpleTrackball::update(GLFWwindow* window) {
	
    if (!ImGui::GetIO().WantCaptureMouse) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            float changeX = (static_cast<float>(x) - m_oldX) * m_sensitivity;
            float changeY = (static_cast<float>(y) - m_oldY) * m_sensitivity;

            m_theta -= changeY;
            if (m_theta < 0.01f) {
                m_theta = 0.01f;
            }
            else if (m_theta > glm::pi<float>() - 0.01f) {
                m_theta = glm::pi<float>() - 0.01f;
            }

            m_phi -= changeX;
            if (m_phi < 0) {
                m_phi += 2 * glm::pi<float>();
            }
            else if (m_phi > 2 * glm::pi<float>()) {
                m_phi -= 2 * glm::pi<float>();
            }
        }

        m_oldX = static_cast<float>(x);
        m_oldY = static_cast<float>(y);

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            m_radius -= 0.1f;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            m_radius += 0.1f;
        }
        if (m_radius < 0.1f) {
            m_radius = 0.1f;
        }

        m_pos.x = m_center.x + m_radius * sin(m_theta) * sin(m_phi);
        m_pos.y = m_center.y + m_radius * cos(m_theta);
        m_pos.z = m_center.z + m_radius * sin(m_theta) * cos(m_phi);

        m_viewMatrix = glm::lookAt(m_pos, m_center, m_up);
    }

}

const glm::mat4& SimpleTrackball::getView() const {
	return m_viewMatrix;
}

glm::mat4& SimpleTrackball::getView() {
    return m_viewMatrix;
}