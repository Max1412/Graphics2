#pragma once

#include <vector>

#include <GL/glew.h>

class Shader
{
public:
	Shader(const std::string& path, GLuint shaderType);
    explicit Shader(GLuint shaderType);
	~Shader();

    /**
     * \brief inits with a given path (loading the shader)
     * \param path relative to SHADERS_PATH
     */
    void init(const std::string& path) const;

    /**
     * \brief inits with the path given in the constructor
     */
    void init() const;

    /**
	 * \brief returns the shader handle
	 * \return shader handle
	 */
	GLuint getHandle() const;

    /**
     * \brief returns the shader type (vertex, fragment, ...)
     * \return shader type
     */
    GLuint getShaderType() const;


private:
	GLuint m_shaderHandle;
    GLuint m_shaderType;
    std::string m_path;

	std::string loadShaderFile(const std::string& fileName) const;

};