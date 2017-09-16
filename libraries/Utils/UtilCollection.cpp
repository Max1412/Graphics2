#include "UtilCollection.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "IO/stb_image_write.h"

#include <ctime>
#include <iostream>
#include <sstream>

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
		GLFWwindow* window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
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
		if constexpr(debugmode) {
            if (glfwGetCurrentContext() != nullptr) {
                GLenum err;
                while ((err = glGetError()) != GL_NO_ERROR) {
                    std::cout << "OpenGL Error: " << err << std::endl;
                    std::cout << "Last error check in function " << function << " at line " << line << std::endl;
                }
            }
		}
	}

	void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
		const GLchar *message, const void *userParam) {
		std::cout << "OpenGL debug callback called!" << '\n';
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
		std::cout << '\n';
		std::cout << "message: " << message << '\n';
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
		std::cout << '\n';
		std::cout << "id: " << id << '\n';
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
		std::cout << '\n';
		std::cout << std::endl;
	}

    void enableDebugCallback() {
        if constexpr(debugmode) {
            glDebugMessageCallback(debugCallback, nullptr);

            // disable notifications and memory info
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0, nullptr, GL_FALSE);

            // enable more severe errors
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_MEDIUM, 0, nullptr, GL_TRUE);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
        }
    }

    void savePNG(std::string name, std::vector<unsigned char>& image, int width, int height) {
        // flip
#pragma omp parallel for
        for (int yi = 0; yi < (height / 2); yi++) {
            for (int xi = 0; xi < width; xi++) {
                unsigned int offset1 = (xi + (yi * width)) * 4;
                unsigned int offset2 = (xi + ((height - 1 - yi) * width)) * 4;
                for (int bi = 0; bi < 4; bi++) {
                    unsigned char byte1 = image[offset1 + bi];
                    unsigned char byte2 = image[offset2 + bi];
                    image[offset1 + bi] = byte2;
                    image[offset2 + bi] = byte1;
                }
            }
        }
        std::stringstream path;
        path << (RESOURCES_PATH) << "../../../" << name << "_" << time(nullptr) << ".png";
        
        int err = stbi_write_png(path.str().c_str(), width, height, 4, image.data(), 4 * width);
        if (err == 0) throw std::runtime_error("error writing image");

    }

    void saveFBOtoFile(std::string name, GLFWwindow* window) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        int width;
        int height;
        glfwGetFramebufferSize(window, &width, &height);
        std::vector<unsigned char> image;
        image.resize(width * height * 4);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image.data());

        //Encode the image
        try {
            std::thread{ savePNG, name, image, width, height }.detach();
        }
        catch (std::runtime_error& ex) {
            std::cout << ex.what() << std::endl;
        }
    }

}
