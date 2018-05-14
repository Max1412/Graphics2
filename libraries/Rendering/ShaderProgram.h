#pragma once

#include <memory>
#include <vector>
#include <utility>
#include <any>

#include <glbinding/gl/gl.h>
using namespace gl;

#include "Shader.h"
#include "Uniform.h"
#include "Utils/UtilCollection.h"
#include "glshader/include/glsp/glsp.hpp"


class ShaderProgram
{
public:
    ShaderProgram(const std::experimental::filesystem::path& vspath, const std::experimental::filesystem::path& fspath, const std::vector<glsp::definition>& definitions = {});
    ShaderProgram(const Shader& shader1, const Shader& shader2);
    ShaderProgram(const std::vector<Shader>& shaders);
    ~ShaderProgram();

    /**
     * \brief adds a shaderto the program
     * \param shader the shader that is to be added
     */
    void addShader(const Shader& shader);

    /**
     * \brief creates a shader program with all shaders from the map
     */
    void createProgram();

    /**
     * \brief links the shader program
     */
    void linkProgram() const;

    /**
     * \brief retruns the shader program handle
     * \return shader program handle
     */
    GLuint getShaderProgramHandle() const;

    /**
     * \brief sets the shader program as the currently used shader program
     */
    void use() const;

    /**
     * \brief swaps current shader of the same kind with the given shader
     * \param shader new shader to be used
     */
    void changeShader(const Shader& shader);

    /**
     * \brief adds an unifrom to the container of uniforms
     * \tparam UniformType the type of the uniform to be added
     * \param uniform the uniform itself
     */
    template <typename UniformType>
    void addUniform(std::shared_ptr<Uniform<UniformType>> uniform);

    /**
     * \brief updates all uniforms depending on their flags
     */
    void updateUniforms() const;

    /**
     * \brief shows a "reload vertex/fragment shader" gui window using imgui
     * \param vshader the vertex shader to be changed/reloaded
     * \param fshader the fragment shader to be changed/reloaded
     * \param name debug name in the GUI
     */
    void showReloadShaderGUI(const std::vector<Shader>& shaders, std::string_view name = "Generic Shaderprogram");

	/**
	* \brief returns "reload vertex/fragment shader" gui content to use with imgui
	* \param vshader the vertex shader to be changed/reloaded
	* \param fshader the fragment shader to be changed/reloaded
	* \param name debug name in the GUI
	*/
	void showReloadShaderGUIContent(const std::vector<Shader>& shaders, std::string_view name = "Generic Shaderprogram");

private:
    GLuint m_shaderProgramHandle;
    std::unordered_map<GLenum, Shader> m_shaderMap;
    bool m_initWithShaders = false;

    std::vector<std::pair<std::any, GLint>> m_anyUniforms;

    /**
    * \brief forces update of all uniforms (ignores flag)
    */
    void forceUpdateUniforms();
};

template <typename UniformType>
void ShaderProgram::addUniform(std::shared_ptr<Uniform<UniformType>> uniform)
{
    GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
    if (location < 0)
        throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
    uniform->registerUniformWithShaderProgram(m_shaderProgramHandle);
    m_anyUniforms.push_back(std::make_pair(std::make_any<std::shared_ptr<Uniform<UniformType>>>(uniform), location));
}
