#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <array>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>


class ShaderProgram
{
public:
	ShaderProgram(std::string vspath, std::string fspath);
	~ShaderProgram();
	
	GLuint getShaderProgramHandle();
	void use();

private:
	GLuint m_shaderProgramHandle;
	std::vector<GLuint> m_shaders;


	std::string ShaderProgram::loadShaderFile(const std::string fileName) const;

	// creates a shader and adds it to the shaders vector
	void createShader(std::string path, GLuint shaderType);


	void createProgram();





};