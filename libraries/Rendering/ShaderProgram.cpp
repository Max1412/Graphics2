#include "ShaderProgram.h"


ShaderProgram::ShaderProgram(std::string vspath, std::string fspath) : m_initWithShaders(true) {
	m_shaders.push_back(Shader(vspath, GL_VERTEX_SHADER));
	m_shaders.push_back(Shader(vspath, GL_VERTEX_SHADER));

	createProgram();
}


ShaderProgram::ShaderProgram(const Shader &shader1, const Shader &shader2) : m_initWithShaders(true) {
	m_shaders.push_back(shader1);
	m_shaders.push_back(shader2);

	createProgram();
}

ShaderProgram::ShaderProgram(std::vector<Shader> shaders) : m_initWithShaders(true) {
	m_shaders = shaders;

	createProgram();
}


void ShaderProgram::del() {
	// delete all shaders
	for(auto shader : m_shaders)
		glDeleteShader(shader.getHandle());

	// delete porgram
	glDeleteProgram(m_shaderProgramHandle);
	util::getGLerror(__LINE__, __FUNCTION__);
}

void ShaderProgram::addShader(const Shader &shader) {
	if (m_initWithShaders) {
		throw std::runtime_error("ShaderProgram was initalized with Shaders, adding later on is not allowed");
	}
	m_shaders.push_back(shader);
}

void ShaderProgram::createProgram() {

	// check if there are shaders in this ShaderProgram
	if (m_shaders.size() == 0) {
		throw std::runtime_error("No shaders in this ShaderProgram! Plase add shaders before calling createProgram()!");
	}


	// create Program and check for errors
	m_shaderProgramHandle = glCreateProgram();
	if (0 == m_shaderProgramHandle)
	{
		throw std::runtime_error("Error creating program.");
	}

	// attach all shaders
	for(auto n : m_shaders)
		glAttachShader(m_shaderProgramHandle, n.getHandle());

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

GLuint ShaderProgram::getShaderProgramHandle() {
	return m_shaderProgramHandle;
}

void ShaderProgram::use() {
	glUseProgram(m_shaderProgramHandle);
	updateUniforms();
}

void ShaderProgram::addUniform(std::shared_ptr<Uniform<glm::mat4>> uniform) {
	use();
	GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
	if(location < 0)
		throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
	m_mat4Uniforms.push_back(std::make_pair(uniform, location));
}

void ShaderProgram::addUniform(std::shared_ptr<Uniform<bool>> uniform) {
    use();
    GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
    if (location < 0)
        throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
    m_boolUniforms.push_back(std::make_pair(uniform, location));
}


void ShaderProgram::updateUniforms() {
	for (auto n : m_mat4Uniforms) {
		glUniformMatrix4fv(n.second, 1, GL_FALSE, glm::value_ptr(n.first->getContent()));
	}
    for (auto n : m_boolUniforms) {
        if (n.first->getContent()) {
            glUniform1i(n.second, 1);
        }
        else {
            glUniform1i(n.second, 0);

        }
    }
}