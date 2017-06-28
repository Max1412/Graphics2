#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>

int main(int argc, char* argv[]) {
	glfwInit();
	GLFWwindow* window = glfwCreateWindow(1600, 980, "Example 1", NULL, NULL);
	glfwMakeContextCurrent(window);
	
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::stringstream ss;
		ss << "Error initializing GLEW: " << glewGetErrorString(err);
		throw std::runtime_error(ss.str());
	}
	glfwDestroyWindow(window);
}
