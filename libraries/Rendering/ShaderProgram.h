#pragma once


#include <memory>
#include <vector>
#include <map>
#include <utility>

#include <GL/glew.h>
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
    ~ShaderProgram();

	
	void addShader(const Shader &shader);

	void createProgram();
    void linkProgram();

	GLuint getShaderProgramHandle();

	void use();

    void changeShader(const Shader &shader);

    void addUniform(std::shared_ptr<Uniform<glm::mat4>> uniform);
    void addUniform(std::shared_ptr<Uniform<glm::vec3>> uniform);
    void addUniform(std::shared_ptr<Uniform<glm::vec2>> uniform);
    void addUniform(std::shared_ptr<Uniform<bool>> uniform);
    void addUniform(std::shared_ptr<Uniform<int>> uniform);
    void addUniform(std::shared_ptr<Uniform<float>> uniform);



	void updateUniforms();
    void forceUpdateUniforms();

    void showReloadShaderGUI(const Shader& vshader, const Shader& fshader);

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



};