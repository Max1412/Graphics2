#pragma once

#include <vector>
#include <thread>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace util
{
	/**
	 * \brief converts a GLubyte* 'char array' to a std::string
	 * \param content input GLubyte* 'char array'
	 * \return std::string with same text as input
	 */
	std::string convertGLubyteToString(const GLubyte* content);

	/**
	 * \brief prints the OpenGL driver/vendor info to the console
	 */
	void printOpenGLInfo();

	/**
	 * \brief sets up a GLFW window/context
	 * \param width window width in pixels
	 * \param height window height in pixels
	 * \param name window name
	 * \return window pointer for later use by glfw functions
	 */
	GLFWwindow* setupGLFWwindow(unsigned int width, unsigned int height, std::string name);

	/**
	 * \brief inits glew
	 */
	void initGLEW();

	/**
	 * \brief queries all available OpenGL extensions
	 * \return vector of extensions as strings
	 */
	std::vector<std::string> getGLExtenstions();

	/**
	 * \brief checks the OpenGL error stack (old way of getting errors)
	 * \param line use __LINE__
	 * \param function use __FUNCTION__
	 */
	void getGLerror(int line, std::string function);

	/**
	 * \brief saves the FBO content to a PNG file (starts a new thread)
	 * \param name output filename
	 * \param window glfw window
	 */
	void saveFBOtoFile(std::string name, GLFWwindow* window);

	/**
	 * \brief enabled OpenGL debug callback (new way of getting errors)
	 */
	void enableDebugCallback();

	static constexpr bool debugmode = true;
}
