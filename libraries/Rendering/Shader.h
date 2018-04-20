#pragma once

#include <vector>
#include <filesystem>

#include <GL/glew.h>
#include "glshader/include/glsp/glsp.hpp"

class Shader
{
public:
    Shader(const std::experimental::filesystem::path& path, GLuint shaderType, const std::vector<glsp::definition>& definitions = {});
    
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
    std::experimental::filesystem::path m_path;

    std::vector<glsp::definition> m_definitions;

    std::string loadShaderFile(const std::experimental::filesystem::path& fileName) const;
};
