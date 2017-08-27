#pragma once


#include <memory>
#include <vector>
#include <map>
#include <utility>
#include <any>

#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.h"
#include "Uniform.h"
#include "Utils/UtilCollection.h"


class ShaderProgram
{
public:
	ShaderProgram(std::string vspath, std::string fspath);
	ShaderProgram(const Shader &shader1, const Shader &shader2);
	ShaderProgram(const std::vector<Shader>& shaders);
    ~ShaderProgram();

	
    /**
	 * \brief adds a shaderto the program
	 * \param shader the shader that is to be added
	 */
	void addShader(const Shader &shader);

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
	void use();

    /**
     * \brief swaps current shader of the same kind with the given shader
     * \param shader new shader to be used
     */
    void changeShader(const Shader &shader);

    void addUniform(std::shared_ptr<Uniform<glm::mat4>> uniform);
    void addUniform(std::shared_ptr<Uniform<glm::vec3>> uniform);
    void addUniform(std::shared_ptr<Uniform<glm::vec2>> uniform);
    void addUniform(std::shared_ptr<Uniform<bool>> uniform);
    void addUniform(std::shared_ptr<Uniform<int>> uniform);
    void addUniform(std::shared_ptr<Uniform<float>> uniform);
    void updateAnyUniforms();

    /**
	 * \brief updates all uniforms depending on their flags
	 */
	void updateUniforms();

    /**
     * \brief forces update of all uniforms (ignores flag)
     */
    void forceUpdateUniforms();

    /**
     * \brief shows a "reload vertex/fragment shader" gui window using imgui
     * \param vshader the vertex shader to be changed/reloaded
     * \param fshader the fragment shader to be changed/reloaded
     */
    void showReloadShaderGUI(const Shader& vshader, const Shader& fshader);

    template<typename UniformType>
    void addAnyUniform(std::shared_ptr<Uniform<UniformType>> uniform)
    {
        GLint location = glGetUniformLocation(m_shaderProgramHandle, uniform->getName().c_str());
        if (location < 0)
            throw std::runtime_error("Uniform " + uniform->getName() + " does not exist");
        m_anyUniforms.push_back(std::make_pair(std::make_any<std::shared_ptr<Uniform<UniformType>>>(uniform), location));
    }

private:
	GLuint m_shaderProgramHandle;
    std::map<GLuint, Shader> m_shaderMap;
	bool m_initWithShaders = false;

    std::vector<std::pair<std::shared_ptr<Uniform<glm::vec3>>, GLint>> m_vec3Uniforms;
    std::vector<std::pair<std::shared_ptr<Uniform<glm::vec2>>, GLint>> m_vec2Uniforms;
	std::vector<std::pair<std::shared_ptr<Uniform<glm::mat4>>, GLint>> m_mat4Uniforms;
    std::vector<std::pair<std::shared_ptr<Uniform<bool>>, GLint>> m_boolUniforms;
    std::vector<std::pair<std::shared_ptr<Uniform<int>>, GLint>> m_intUniforms;
    std::vector<std::pair<std::shared_ptr<Uniform<float>>, GLint>> m_floatUniforms;

    std::vector<std::pair<std::any, GLint>> m_anyUniforms;



};