#pragma once

#include <vector>
#include <thread>
#include <experimental/filesystem>

#include <glbinding/gl/gl.h>
using namespace gl;
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
     * \brief inits the graphics API
     */
    void initGL();

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

    /** 
     * \brief Calls the provided function and returns the number of milliseconds 
     * that it takes to call that function.
     * \param f function to call
     */
    template <class Function>
    double timeCall(Function&& f)
    {
        auto begin = glfwGetTime();
        f();
        glFinish();
        return (glfwGetTime() - begin) * 1000;
    }

    /**
    * \brief Container for overloaded functions/lambdas
    */
    template<class... Fs>
    struct overload : Fs...
    {
        overload(Fs&&... fs)
            : Fs(std::move(fs))...
        {}
    };

    /**
    * \brief Makes an overloaded function/lambda object
    * \param fs The functions to be overloaded
    */
    auto const make_overload = [](auto... fs)
    {
        return overload<decltype(fs)...>{std::move(fs)...};
    };


    static const std::experimental::filesystem::path gs_shaderPath = std::experimental::filesystem::current_path().parent_path().parent_path().append("shaders");
    static const std::experimental::filesystem::path gs_resourcesPath = std::experimental::filesystem::current_path().parent_path().parent_path().append("resources");

#ifdef _DEBUG
    static constexpr bool debugmode = true;
#else
	static constexpr bool debugmode = false;
#endif

}
