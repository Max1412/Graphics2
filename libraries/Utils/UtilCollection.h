#pragma once

#include <vector>
#include <thread>

#include <GL/glew.h>
#include <GLFW/glfw3.h>


namespace util
{
	std::string convertGLubyteToString(const GLubyte* content);

	void printOpenGLInfo();

	GLFWwindow* setupGLFWwindow(unsigned int width, unsigned int height, std::string name);

	void initGLEW();

	std::vector<std::string> getGLExtenstions();

	void getGLerror(int line, std::string function);

    void saveFBOtoFile(std::string name, GLFWwindow* window);

	void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar * message, void * param);

	void enableDebugCallback();

	const bool debugmode = true;
}