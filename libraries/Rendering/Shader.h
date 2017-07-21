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
	Shader(const std::string& path, GLuint shaderType);
    Shader::Shader(GLuint shaderType);
	~Shader();

    void Shader::init(const std::string& path);
    void Shader::init();


	GLuint getHandle() const;
    GLuint getShaderType() const;


private:
	GLuint m_shaderHandle;
    GLuint m_shaderType;
    std::string m_path;

	std::string Shader::loadShaderFile(const std::string& fileName) const;

};