#define GLEW_STATIC
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
	
	const GLubyte *renderer = glGetString(GL_RENDERER);
	const GLubyte *vendor = glGetString(GL_VENDOR);
	const GLubyte *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
	
	std::cout << renderer << std::endl;
	std::cout << vendor << std::endl;
	std::cout << version << std::endl;
	std::cout << glslVersion << std::endl;
	
	while (!glfwWindowShouldClose(window)) {


	}


	glfwDestroyWindow(window);
}
