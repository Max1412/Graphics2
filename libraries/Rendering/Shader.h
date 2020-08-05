#pragma once

#include <vector>
#include <filesystem>

#include <glbinding/gl/gl.h>
using namespace gl;
#include "glshader/include/glsp/glsp.hpp"

class Shader
{
public:
    Shader(const std::filesystem::path& path, GLenum shaderType, const std::vector<glsp::definition>& definitions = {});
    
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
    GLenum getShaderType() const;

private:
    GLuint m_shaderHandle;
    GLenum m_shaderType;
    std::filesystem::path m_path;

    std::vector<glsp::definition> m_definitions;

    std::string loadShaderFile(const std::filesystem::path& fileName) const;
};
