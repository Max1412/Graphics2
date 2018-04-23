#include "Shader.h"

#include <array>
#include <iostream>
#include <fstream>
#include <string>

#include "Utils/Timer.h"


Shader::Shader(const std::experimental::filesystem::path& path, GLenum shaderType, const std::vector<glsp::definition>& definitions) : m_shaderType(shaderType), m_path(path), m_definitions(definitions)
{
    // create shader and check for errors
    m_shaderHandle = glCreateShader(shaderType);
    if (0 == m_shaderHandle)
    {
        throw std::runtime_error("Error creating shader.");
    }
    init();
}

void Shader::init() const
{
    if (m_path.empty())
    {
        throw std::runtime_error("No path given");
    }
    // load shader file and use it
    auto shaderCode = loadShaderFile(std::experimental::filesystem::path(util::gs_shaderPath) / m_path);
    std::array<const GLchar*, 1> codeArray{shaderCode.c_str()};
    glShaderSource(m_shaderHandle, 1, codeArray.data(), nullptr);

    // compile shader
    glCompileShader(m_shaderHandle);

    // check of compilation was succesful, print log if not
    GLint result;
    glGetShaderiv(m_shaderHandle, GL_COMPILE_STATUS, &result);
    if (GL_FALSE == result)
    {
        GLint logLen;
        glGetShaderiv(m_shaderHandle, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0)
        {
            std::string log;
            log.resize(logLen);
            GLsizei written;
            glGetShaderInfoLog(m_shaderHandle, logLen, &written, &log[0]);
            std::cout << "Shader log: " << log << std::endl;
        }
        throw std::runtime_error("Shader compilation failed");
    }
    util::getGLerror(__LINE__, __FUNCTION__);
}

GLuint Shader::getHandle() const
{
    return m_shaderHandle;
}

GLenum Shader::getShaderType() const
{
    return m_shaderType;
}

std::string Shader::loadShaderFile(const std::experimental::filesystem::path& fileName) const
{
    return glsp::preprocess_file(fileName, { util::gs_shaderPath }, m_definitions).contents;
}
