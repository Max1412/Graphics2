#include "Shader.h"

Shader::Shader(std::string path, GLuint shaderType)
{
	// create shader and check for errors
	m_shaderHandle = glCreateShader(shaderType);
	if (0 == m_shaderHandle)
	{
		throw std::runtime_error("Error creating shader.");
	}

	// load shader file and use it
	std::string shaderCode = loadShaderFile(SHADERS_PATH + std::string("/") + path);
	std::array<const GLchar*, 1> codeArray{ shaderCode.c_str() };
	glShaderSource(m_shaderHandle, 1, codeArray.data(), NULL);

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
}

Shader::~Shader() {

}

GLuint Shader::getHandle() {
	return m_shaderHandle;
}

std::string Shader::loadShaderFile(const std::string fileName) const
{
	std::string fileContent;
	std::string line;

	// open file and concatenate input
	std::ifstream file(fileName);
	if (file.is_open()) {
		while (!file.eof()) {
			getline(file, line);
			fileContent += line + "\n";
		}
		file.close();
		std::cout << "SUCCESS: Opened file " << fileName << std::endl;
	}
	else
		throw std::runtime_error("ERROR: Unable to open file " + fileName);

	// return file as string
	return fileContent;
}