#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>

#include "Utils/UtilCollection.h"

std::string GLubyteToString(const GLubyte* content) {
	return std::string(reinterpret_cast<const char*>(content));

}

int main(int argc, char* argv[]) {
	// init glfw, open window, manage context
	GLFWwindow* window = util::setupGLFWwindow(1600, 980, "Example 1");
	
	// init glew and check for errors
	util::initGLEW();

	// print OpenGL info
	util::printOpenGLInfo();

	// get list of OpenGL extensions (can be searched later if needed)
	std::vector<std::string> extensions = util::getGLExtenstions();

	
	// render loop
	while (!glfwWindowShouldClose(window)) {


	}

	// close window
	glfwDestroyWindow(window);
}
