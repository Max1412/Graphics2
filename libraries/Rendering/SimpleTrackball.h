#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class SimpleTrackball {
public:
	SimpleTrackball(int width, int height, float radius);

	void update(GLFWwindow* window);
	glm::mat4 getView();

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