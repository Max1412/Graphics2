#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <array>
#include <vector>
#include <utility>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.h"
#include "Uniform.h"
#include "Utils/UtilCollection.h"


class ShaderProgram
{
public:
	ShaderProgram();
	ShaderProgram(std::string vspath, std::string fspath);
	ShaderProgram(const Shader &shader1, const Shader &shader2);
	ShaderProgram(const std::vector<Shader>& shaders);

	void del();
	
	void addShader(const Shader &shader);

	void createProgram();

	GLuint getShaderProgramHandle();

	void use();

	void addUniform(std::shared_ptr<Uniform<glm::mat4>> uniform);
    void addUniform(std::shared_ptr<Uniform<bool>> uniform);
    void addUniform(std::shared_ptr<Uniform<int>> uniform);


	void updateUniforms();

private:
	GLuint m_shaderProgramHandle;
	std::vector<Shader> m_shaders;
	bool m_initWithShaders = false;

	std::vector<std::pair<std::shared_ptr<Uniform<glm::mat4>>, GLint>> m_mat4Uniforms;
    std::vector<std::pair<std::shared_ptr<Uniform<bool>>, GLint>> m_boolUniforms;
    std::vector<std::pair<std::shared_ptr<Uniform<int>>, GLint>> m_intUniforms;


};