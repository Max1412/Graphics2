#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(std::string vspath, std::string fspath) : m_initWithShaders(true) {
    Shader vs(vspath, GL_VERTEX_SHADER);
    Shader fs(fspath, GL_FRAGMENT_SHADER);

    m_shaderMap.insert(std::make_pair(vs.getShaderType(), vs));
    m_shaderMap.insert(std::make_pair(fs.getShaderType(), fs));

    createProgram();
    linkProgram();
}

ShaderProgram::ShaderProgram(const Shader &shader1, const Shader &shader2) : m_initWithShaders(true) {
    m_shaderMap.insert(std::make_pair(shader1.getShaderType(), shader1));
    m_shaderMap.insert(std::make_pair(shader2.getShaderType(), shader2));

	createProgram();
    linkProgram();
}

ShaderProgram::ShaderProgram(const std::vector<Shader>& shaders) : m_initWithShaders(true) {
    for(auto n : shaders)
        m_shaderMap.insert(std::make_pair(n.getShaderType(), n));

	createProgram();
    linkProgram();
}

void ShaderProgram::changeShader(const Shader &shader) {
    // find out which shader has to be changed and detach it
    auto search = m_shaderMap.find(shader.getShaderType());
    if(search == m_shaderMap.end())
    {
        throw std::runtime_error("No matching shader found");
    }
    glDetachShader(m_shaderProgramHandle, search->second.getHandle());

    // insert new shader into map, attach it and relink
    glAttachShader(m_shaderProgramHandle, shader.getHandle());
    try
    {
        linkProgram();
    } catch(std::runtime_error &err)
    {
        std::cout << "linking failed, rolling back to the old shader" << std::endl;
        std::cout << err.what() << std::endl;
        glDetachShader(m_shaderProgramHandle, shader.getHandle());
        glAttachShader(m_shaderProgramHandle, search->second.getHandle());
        linkProgram();
    }
    // insert new shader into map when everything worked
    m_shaderMap.insert(std::make_pair(shader.getShaderType(), shader));

}


void ShaderProgram::del() {
	// delete all shaders
	for(auto shaderPair : m_shaderMap)
		glDeleteShader(shaderPair.second.getHandle());

	// delete porgram
	glDeleteProgram(m_shaderProgramHandle);
	util::getGLerror(__LINE__, __FUNCTION__);
}

void ShaderProgram::addShader(const Shader &shader) {
	if (m_initWithShaders) {
		throw std::runtime_error("ShaderProgram was initalized with Shaders, adding later on is not allowed");
	}
    m_shaderMap.insert(std::make_pair(shader.getShaderType(), shader));
}

void ShaderProgram::createProgram() {

	// check if there are shaders in this ShaderProgram
	if (m_shaderMap.size() == 0) {
		throw std::runtime_error("No shaders in this ShaderProgram! Please add shaders before calling createProgram()!");
	}


	// create Program and check for errors
	m_shaderProgramHandle = glCreateProgram();
	if (0 == m_shaderProgramHandle)
	{
		throw std::runtime_error("Error creating program.");
	}

	// attach all shaders
	for(auto n : m_shaderMap)
		glAttachShader(m_shaderProgramHandle, n.second.getHandle());
}

void ShaderProgram::linkProgram()
{
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
        util::getGLerror(__LINE__, __FUNCTION__);
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

void ShaderProgram::addUniform(std::shared_ptr<Uniform<glm::vec3>> uniform) {
    use();
    GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
    if (location < 0)
        throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
    m_vec3Uniforms.push_back(std::make_pair(uniform, location));
}

void ShaderProgram::addUniform(std::shared_ptr<Uniform<bool>> uniform) {
    use();
    GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
    if (location < 0)
        throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
    m_boolUniforms.push_back(std::make_pair(uniform, location));
}

void ShaderProgram::addUniform(std::shared_ptr<Uniform<int>> uniform) {
    use();
    GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
    if (location < 0)
        throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
    m_intUniforms.push_back(std::make_pair(uniform, location));
}


void ShaderProgram::updateUniforms() {
    for (auto n : m_mat4Uniforms) {
        if (n.first->getChangeFlag()) {
            glUniformMatrix4fv(n.second, 1, GL_FALSE, glm::value_ptr(n.first->getContent()));
            n.first->hasBeenUpdated();
        }
    }
    for (auto n : m_boolUniforms) {
        if (n.first->getChangeFlag()) {
            if (n.first->getContent()) {
                glUniform1i(n.second, 1);
            }
            else {
                glUniform1i(n.second, 0);
            }
            n.first->hasBeenUpdated();
        }
    }
    for (auto n : m_intUniforms) {
        if (n.first->getChangeFlag()) {
            glUniform1i(n.second, n.first->getContent());
            n.first->hasBeenUpdated();
        }
    }
    for (auto n : m_vec3Uniforms) {
        if (n.first->getChangeFlag()) {
            glUniform3fv(n.second, 1, glm::value_ptr(n.first->getContent()));
            n.first->hasBeenUpdated();
        }
    }
}

void ShaderProgram::forceUpdateUniforms() {
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

        for (auto n : m_intUniforms) {
            glUniform1i(n.second, n.first->getContent());
        }
        for (auto n : m_vec3Uniforms) {
            glUniform3fv(n.second, 1, glm::value_ptr(n.first->getContent()));
        }
}