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
				std::cerr << "OpenGL Error: " << err << std::endl;
				std::cerr << "Last error check in function " << function << " at line " << line << std::endl;
			}
		}
	}
	void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
		const GLchar *message, const void *userParam) {
		std::cout << "OpenGL debug callback called!" << std::endl;
		std::cout << "Source: ";
		switch (source) {
		case GL_DEBUG_SOURCE_API:
			std::cout << "API Call";
			break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
			std::cout << "Window system";
			break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			std::cout << "Third party application";
			break;
		case GL_DEBUG_SOURCE_APPLICATION:
			std::cout << "This application";
			break;
		case GL_DEBUG_SOURCE_OTHER:
			std::cout << "Some other source";
			break;
		}
		std::cout << std::endl;
		std::cout << "message: " << message << std::endl;
		std::cout << "type: ";
		// converting GLenums is tedious :(
		switch (type) {
		case GL_DEBUG_TYPE_ERROR:
			std::cout << "ERROR";
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			std::cout << "DEPRECATED_BEHAVIOR";
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			std::cout << "UNDEFINED_BEHAVIOR";
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			std::cout << "PORTABILITY";
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			std::cout << "PERFORMANCE";
			break;
		case GL_DEBUG_TYPE_MARKER:
			std::cout << "Annotation (MARKER)";
			break;
		case GL_DEBUG_TYPE_PUSH_GROUP:
			std::cout << "Debug push group";
			break;
		case GL_DEBUG_TYPE_POP_GROUP:
			std::cout << "Debug pop group";
			break;
		case GL_DEBUG_TYPE_OTHER:
			std::cout << "OTHER";
			break;
		}
		std::cout << std::endl;
		std::cout << "id: " << id << std::endl;
		std::cout << "severity: ";
		switch (severity) {
		case GL_DEBUG_SEVERITY_LOW:
			std::cout << "LOW";
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			std::cout << "MEDIUM";
			break;
		case GL_DEBUG_SEVERITY_HIGH:
			std::cout << "HIGH";
			break;
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			std::cout << "NOTIFICATION";
			break;
		}
		std::cout << std::endl;
		std::cout << std::endl;
	}

	void enableDebugCallback() {
		glDebugMessageCallback(debugCallback, NULL);
		// TODO set ifs/elses for enabling notification, low, high..
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_FALSE);
	}

}
