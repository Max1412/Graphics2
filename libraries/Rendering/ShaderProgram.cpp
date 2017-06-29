#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(std::string vspath, std::string fspath)
{
	// create shaders
	createShader(vspath, GL_VERTEX_SHADER);
	createShader(fspath, GL_FRAGMENT_SHADER);

	// create shader program
	createProgram();
}

ShaderProgram::~ShaderProgram() {
	// delete all shaders
	for(auto shader : m_shaders)
		glDeleteShader(shader);

	// delete porgram
	glDeleteProgram(m_shaderProgramHandle);
}

void ShaderProgram::createProgram() {
	// create Program and check for errors
	m_shaderProgramHandle = glCreateProgram();
	if (0 == m_shaderProgramHandle)
	{
		throw std::runtime_error("Error creating program.");
	}

	// attach all shaders
	for(auto n : m_shaders)
		glAttachShader(m_shaderProgramHandle, n);

	// link program
	glLinkProgram(m_shaderProgramHandle);

	// check if linking was succesful, print log if not
	GLint status;
	glGetProgramiv(m_shaderProgramHandle, GL_LINK_STATUS, &status);
	if (GL_FALSE == status) {
		GLint logLen;
		glGetProgramiv(m_shaderProgramHandle, GL_INFO_LOG_LENGTH,
			&logLen);
		if (logLen > 0)
		{
			std::string log;
			log.resize(logLen);
			GLsizei written;
			glGetShaderInfoLog(m_shaderProgramHandle, logLen, &written, &log[0]);
			std::cout << "Program log: " << log << std::endl;
		}
		throw std::runtime_error("Failed to link shader program!\n");

	}
}

void ShaderProgram::createShader(std::string path, GLuint shaderType) {

	// create shader and check for errors
	GLuint vertexShader = glCreateShader(shaderType);
	if (0 == vertexShader)
	{
		throw std::runtime_error("Error creating shader.");
	}

	// load shader file and use it
	std::string shaderCode = loadShaderFile(SHADERS_PATH + path);
	std::array<const GLchar*, 1> codeArray{ shaderCode.c_str() };
	glShaderSource(vertexShader, 1, codeArray.data(), NULL);

	// compile shader
	glCompileShader(vertexShader);

	// check of compilation was succesful, print log if not
	GLint result;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &result);
	if (GL_FALSE == result)
	{
		GLint logLen;
		glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0)
		{
			std::string log;
			log.resize(logLen);
			GLsizei written;
			glGetShaderInfoLog(vertexShader, logLen, &written, &log[0]);
			std::cout << "Shader log: " << log << std::endl;
		}
		throw std::runtime_error("Shader compilation failed");
	}

	// add shaders to shaders list
	m_shaders.push_back(vertexShader);

}

std::string ShaderProgram::loadShaderFile(const std::string fileName) const
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

GLuint ShaderProgram::getShaderProgramHandle() {
	return m_shaderProgramHandle;
}

void ShaderProgram::use() {
	glUseProgram(m_shaderProgramHandle);
}