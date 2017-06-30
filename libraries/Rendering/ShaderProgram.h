#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <array>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Shader.h"


class ShaderProgram
{
public:
	ShaderProgram();
	ShaderProgram(std::string vspath, std::string fspath);
	ShaderProgram(Shader &shader1, Shader &shader2);
	ShaderProgram(std::vector<Shader> shaders);
	~ShaderProgram();
	
	void addShader(Shader &shader);

	void createProgram();

	GLuint getShaderProgramHandle();

	void use();

private:
	GLuint m_shaderProgramHandle;
	std::vector<Shader> m_shaders;
	bool m_initWithShaders = false;

};