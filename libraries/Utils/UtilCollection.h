#pragma once

#include <string>
#include <sstream>
#include <exception>
#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>



namespace util
{
	std::string convertGLubyteToString(const GLubyte* content);

	void printOpenGLInfo();

	GLFWwindow* setupGLFWwindow(unsigned int width, unsigned int height, std::string name);

	void initGLEW();

	std::vector<std::string> getGLExtenstions();
}