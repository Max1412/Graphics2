#include "UtilCollection.h"

namespace util
{
	std::string convertGLubyteToString(const GLubyte* content) {
		return std::string(reinterpret_cast<const char*>(content));
	}

	void printOpenGLInfo() {
		std::cout << "Renderer: " << convertGLubyteToString(glGetString(GL_RENDERER)) << std::endl;
		std::cout << "Vendor: " << convertGLubyteToString(glGetString(GL_VENDOR)) << std::endl;
		std::cout << "Version: " << convertGLubyteToString(glGetString(GL_VERSION)) << std::endl;
		std::cout << "Shading Language Version: " << convertGLubyteToString(glGetString(GL_SHADING_LANGUAGE_VERSION)) << std::endl;
	}

	GLFWwindow* setupGLFWwindow(unsigned int width, unsigned int height, std::string name) {
		glfwInit();
		GLFWwindow* window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
		glfwMakeContextCurrent(window);
		return window;
	}

	void initGLEW() {
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			std::stringstream ss;
			ss << "Error initializing GLEW: " << glewGetErrorString(err);
			throw std::runtime_error(ss.str());
		}
	}

	std::vector<std::string> getGLExtenstions() {
		GLint nExtensions;
		glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);
		std::vector<std::string> extenstions;
		extenstions.reserve(nExtensions);
		for (int i = 0; i < nExtensions; i++) {
			extenstions.push_back(std::string(convertGLubyteToString(glGetStringi(GL_EXTENSIONS, i))));
		}
		return extenstions;
	}

	void getGLerror(int line, std::string function) {
		if(debugmode) {
			GLenum err;
			while ((err = glGetError()) != GL_NO_ERROR) {
				std::cerr << "OpenGL Error: " << err << " in function " << function << " at line " << line << std::endl;
			}
		}
	}

}
