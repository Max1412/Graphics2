#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <array>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>


class Shader
{
public:
	Shader(std::string path, GLuint shaderType);
	~Shader();

	GLuint getHandle();

private:
	GLuint m_shaderHandle;

	std::string Shader::loadShaderFile(const std::string fileName) const;

};