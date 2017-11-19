#include "Shader.h"

#include <array>
#include <iostream>
#include <fstream>
#include <string>

#include "Utils/Timer.h"

Shader::Shader(const std::experimental::filesystem::path& path, GLuint shaderType) : m_shaderType(shaderType), m_path(path)
{
	// create shader and check for errors
	m_shaderHandle = glCreateShader(shaderType);
	if (0 == m_shaderHandle)
	{
		throw std::runtime_error("Error creating shader.");
	}
    init();
}

Shader::Shader(GLuint shaderType) : m_shaderType(shaderType)
{
    // create shader and check for errors
    m_shaderHandle = glCreateShader(shaderType);
    if (0 == m_shaderHandle)
    {
        throw std::runtime_error("Error creating shader.");
    }
}

void Shader::init(const std::experimental::filesystem::path& path) const
{
    // load shader file and use it
    auto shaderCode = loadShaderFile(std::experimental::filesystem::path(SHADERS_PATH) /= path);
    std::array<const GLchar*, 1> codeArray{ shaderCode.c_str() };
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


void Shader::init() const
{
    if(m_path.empty())
    {
        throw std::runtime_error("No path given");
    }
    // load shader file and use it
    auto shaderCode = loadShaderFile(std::experimental::filesystem::path(SHADERS_PATH) /= m_path);
    std::array<const GLchar*, 1> codeArray{ shaderCode.c_str() };
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

Shader::~Shader() {

}

GLuint Shader::getHandle() const {
	return m_shaderHandle;
}

GLuint Shader::getShaderType() const {
    return m_shaderType;
}

std::string Shader::loadShaderFile(const std::experimental::filesystem::path& fileName) const {
	std::string fileContent;
	std::string line;

	// open file and concatenate input
	std::ifstream file(fileName);
	if (file.is_open()) {
		while (!file.eof()) {
			getline(file, line);
            if(line.substr(0, 8) == "#include")
            {
                line = loadShaderFile(fileName.parent_path().string() + "/" + line.substr(10, line.size() - 11));
            }
			fileContent += line + '\n';
		}
		file.close();
		std::cout << "SUCCESS: Opened file " << fileName << std::endl;
	}
	else
		throw std::runtime_error("ERROR: Unable to open file " + fileName.string());

	// return file as string
	return fileContent;
}
