#include "ShaderProgram.h"

#include "imgui/imgui.h"
#include <iostream>
#include <typeinfo>
#include <glm/gtc/type_ptr.hpp>

ShaderProgram::ShaderProgram(const std::experimental::filesystem::path& vspath, const std::experimental::filesystem::path& fspath, const std::vector<glsp::definition>& definitions)
    : m_initWithShaders(true)
{    
    Shader vs(vspath, GL_VERTEX_SHADER, definitions);
    Shader fs(fspath, GL_FRAGMENT_SHADER, definitions);

    m_shaderMap.insert(std::make_pair(vs.getShaderType(), vs));
    m_shaderMap.insert(std::make_pair(fs.getShaderType(), fs));

    createProgram();
    linkProgram();
}

ShaderProgram::ShaderProgram(const Shader& shader1, const Shader& shader2) : m_initWithShaders(true)
{
    m_shaderMap.insert(std::make_pair(shader1.getShaderType(), shader1));
    m_shaderMap.insert(std::make_pair(shader2.getShaderType(), shader2));

    createProgram();
    linkProgram();
}

ShaderProgram::ShaderProgram(const std::vector<Shader>& shaders) : m_initWithShaders(true)
{
    for (const auto& n : shaders)
        m_shaderMap.insert(std::make_pair(n.getShaderType(), n));

    createProgram();
    linkProgram();
}

void ShaderProgram::changeShader(const Shader& shader)
{
    // find out which shader has to be changed and detach it
    const auto search = m_shaderMap.find(shader.getShaderType());
    if (search == m_shaderMap.end())
    {
        throw std::runtime_error("No matching shader found");
    }
    glDetachShader(m_shaderProgramHandle, search->second.getHandle());

    // insert new shader into map, attach it and relink
    glAttachShader(m_shaderProgramHandle, shader.getHandle());
    try
    {
        linkProgram();
    }
    catch (std::runtime_error& err)
    {
        std::cout << "linking failed, rolling back to the old shader" << std::endl;
        std::cout << err.what() << std::endl;
        glDetachShader(m_shaderProgramHandle, shader.getHandle());
        glAttachShader(m_shaderProgramHandle, search->second.getHandle());
        linkProgram();
    }
    // insert new shader into map when everything worked
    m_shaderMap.insert(std::make_pair(shader.getShaderType(), shader));
}

ShaderProgram::~ShaderProgram()
{
    if (glfwGetCurrentContext() != nullptr)
    {
        // delete all shaders
        for (const auto& shaderPair : m_shaderMap)
        glDeleteShader(shaderPair.second.getHandle());

        // delete porgram
        glDeleteProgram(m_shaderProgramHandle);
    }
    util::getGLerror(__LINE__, __FUNCTION__);
}

void ShaderProgram::addShader(const Shader& shader)
{
    if (m_initWithShaders)
    {
        throw std::runtime_error("ShaderProgram was initalized with Shaders, adding later on is not allowed");
    }
    m_shaderMap.insert(std::make_pair(shader.getShaderType(), shader));
}

void ShaderProgram::createProgram()
{
    // check if there are shaders in this ShaderProgram
    if (m_shaderMap.empty())
    {
        throw std::runtime_error("No shaders in this ShaderProgram! Please add shaders before calling createProgram()!");
    }

    // create Program and check for errors
    m_shaderProgramHandle = glCreateProgram();
    if (0 == m_shaderProgramHandle)
    {
        throw std::runtime_error("Error creating program.");
    }

    // attach all shaders
    for (const auto& n : m_shaderMap)
    glAttachShader(m_shaderProgramHandle, n.second.getHandle());
}

void ShaderProgram::linkProgram() const
{
    // link program
    glLinkProgram(m_shaderProgramHandle);

    // check if linking was succesful, print log if not
    GLint status;
    glGetProgramiv(m_shaderProgramHandle, GL_LINK_STATUS, &status);
    if (GL_FALSE == status)
    {
        GLint logLen;
        glGetProgramiv(m_shaderProgramHandle, GL_INFO_LOG_LENGTH,
                       &logLen);
        if (logLen > 0)
        {
            std::string log;
            log.resize(logLen);
            GLsizei written;
            glGetProgramInfoLog(m_shaderProgramHandle, logLen, &written, &log[0]);
            std::cout << "Program log: " << log << std::endl;
        }
        util::getGLerror(__LINE__, __FUNCTION__);
        throw std::runtime_error("Failed to link shader program!\n");
    }
}

GLuint ShaderProgram::getShaderProgramHandle() const
{
    return m_shaderProgramHandle;
}

void ShaderProgram::use() const
{
    glUseProgram(m_shaderProgramHandle);
    updateUniforms();
}

void ShaderProgram::updateUniforms() const
{
    for (auto&& n : m_anyUniforms)
    {
        // case int
        if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<int>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<int>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
            )
            {
                glProgramUniform1i(m_shaderProgramHandle, n.second, a->getContent());
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case float
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<float>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<float>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
            )
            {
                glProgramUniform1f(m_shaderProgramHandle, n.second, a->getContent());
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case bool
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<bool>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<bool>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
            )
            {
                if (a->getContent())
                {
                    glProgramUniform1i(m_shaderProgramHandle, n.second, 1);
                }
                else
                {
                    glProgramUniform1i(m_shaderProgramHandle, n.second, 0);
                }
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case mat4
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::mat4>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<glm::mat4>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
            )
            {
                glProgramUniformMatrix4fv(m_shaderProgramHandle, n.second, 1, GL_FALSE, glm::value_ptr(a->getContent()));
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case vec3
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::vec3>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<glm::vec3>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
            )
            {
                glProgramUniform3fv(m_shaderProgramHandle, n.second, 1, glm::value_ptr(a->getContent()));
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case vec2
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::vec2>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<glm::vec2>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
            )
            {
                glProgramUniform2fv(m_shaderProgramHandle, n.second, 1, glm::value_ptr(a->getContent()));
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case uvec3
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::uvec3>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<glm::uvec3>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
                )
            {
                glProgramUniform3uiv(m_shaderProgramHandle, n.second, 1, glm::value_ptr(a->getContent()));
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        // case ivec3
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::ivec3>>).hash_code())
        {
            if (auto a = std::any_cast<std::shared_ptr<Uniform<glm::ivec3>>>(n.first);
            a->getChangeFlag(m_shaderProgramHandle)
                )
            {
                glProgramUniform3iv(m_shaderProgramHandle, n.second, 1, glm::value_ptr(a->getContent()));
                a->hasBeenUpdated(m_shaderProgramHandle);
            }
        }
        else
        {
            throw std::runtime_error("Uniform type not supported yet.");
        }
    }
}

void ShaderProgram::forceUpdateUniforms()
{
    for (auto&& n : m_anyUniforms)
    {
        // case int
        if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<int>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<int>>>(n.first);
            glProgramUniform1i(m_shaderProgramHandle, n.second, a->getContent());
        }
        // case float
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<float>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<float>>>(n.first);
            glProgramUniform1f(m_shaderProgramHandle, n.second, a->getContent());
        }
        // case bool
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<bool>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<bool>>>(n.first);
            if (a->getContent())
            {
                glProgramUniform1i(m_shaderProgramHandle, n.second, 1);
            }
            else
            {
                glProgramUniform1i(m_shaderProgramHandle, n.second, 0);
            }
        }
        // case mat4
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::mat4>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<glm::mat4>>>(n.first);
            glProgramUniformMatrix4fv(m_shaderProgramHandle, n.second, 1, GL_FALSE, value_ptr(a->getContent()));
        }
        // case vec3
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::vec3>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<glm::vec3>>>(n.first);
            glProgramUniform3fv(m_shaderProgramHandle, n.second, 1, value_ptr(a->getContent()));
        }
        // case vec2
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::vec2>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<glm::vec2>>>(n.first);
            glProgramUniform2fv(m_shaderProgramHandle, n.second, 1, value_ptr(a->getContent()));
        }
        // case uvec3
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::uvec3>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<glm::uvec3>>>(n.first);
            glProgramUniform3uiv(m_shaderProgramHandle, n.second, 1, value_ptr(a->getContent()));
        }
        // case ivec3
        else if (n.first.type().hash_code() == typeid(std::shared_ptr<Uniform<glm::ivec3>>).hash_code())
        {
            const auto a = std::any_cast<std::shared_ptr<Uniform<glm::ivec3>>>(n.first);
            glProgramUniform3iv(m_shaderProgramHandle, n.second, 1, value_ptr(a->getContent()));
        }
        else
        {
            throw std::runtime_error("Uniform type not supported yet.");
        }
        util::getGLerror(__LINE__, __FUNCTION__);
    }
}

void ShaderProgram::showReloadShaderGUI(const std::vector<Shader>& shaders, std::string_view name)
{
    ImGui::SetNextWindowSize(ImVec2(100, 100), ImGuiSetCond_FirstUseEver);
    ImGui::Begin(name.data());
    for (const Shader& shader : shaders) {
        std::stringstream ss;
        ss << "Reload: " << shader.getShaderType();
        if (ImGui::Button(ss.str().c_str()))
        {
            try
            {
                shader.init();
                changeShader(shader);
                use();
                forceUpdateUniforms();
            }
            catch (std::runtime_error& err)
            {
                std::cout << "Shader could not be loaded, not using it" << std::endl;
                std::cout << err.what() << std::endl;
            }
        }
    }
    ImGui::End();
}
