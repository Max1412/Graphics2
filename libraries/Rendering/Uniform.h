#pragma once

#include <string>
#include <glbinding/gl/gl.h>
using namespace gl;
#include <unordered_map>

template <typename T>
class Uniform
{
public:
    Uniform(const std::string& name, T content) :
        m_name(name), m_content(content)
    {
    };

    void registerUniformWithShaderProgram(GLuint);

    /**
     * \brief returns the uniform name
     * \return uniform name
     */
    const std::string& getName() const;

    /**
     * \brief returns the current content
     * \return current content
     */
    T getContent() const;

    /**
     * \brief returns the 'content-has-been-changed'-flag
     * \return change flag
     */
    bool getChangeFlag(GLuint shaderProgramHandle) const;

    /**
     * \brief sets the (cpu-sided) content and the change flag
     * \param content 
     */
    void setContent(const T& content);

    /**
    * \brief resets all change flags
    */
    void hasBeenUpdated(GLuint shaderProgramHandle);

private:
    std::string m_name;
    T m_content;

    // holds the shader program handles and the flag if the uniform is up-to-date in the corresponding shaderprogram
    std::unordered_map<GLuint, bool> m_associatedShaderProgramUpdatedFlags;
};

template <typename T>
void Uniform<T>::registerUniformWithShaderProgram(const GLuint shaderProgramHandle)
{
    m_associatedShaderProgramUpdatedFlags.insert({shaderProgramHandle, true});
}

template <typename T>
bool Uniform<T>::getChangeFlag(const GLuint shaderProgramHandle) const
{
    return m_associatedShaderProgramUpdatedFlags.at(shaderProgramHandle);
}

template <typename T>
const std::string& Uniform<T>::getName() const
{
    return m_name;
}

template <typename T>
void Uniform<T>::hasBeenUpdated(const GLuint shaderProgramHandle)
{
    m_associatedShaderProgramUpdatedFlags.at(shaderProgramHandle) = false;
}

template <typename T>
T Uniform<T>::getContent() const
{
    return m_content;
}

template <typename T>
void Uniform<T>::setContent(const T& content)
{
    for (auto& flags : m_associatedShaderProgramUpdatedFlags)
        flags.second = true;
    m_content = content;
}
