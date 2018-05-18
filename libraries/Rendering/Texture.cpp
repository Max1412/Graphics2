#include "Texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "Utils/UtilCollection.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.inl>
#include <GLFW/glfw3.h>
#include <iostream>

Texture::Texture(GLenum target, GLenum minFilter, GLenum maxFilter)
{
    glCreateTextures(target, 1, &m_name);
    glTextureParameteri(m_name, GL_TEXTURE_MIN_FILTER, minFilter);
    glTextureParameteri(m_name, GL_TEXTURE_MAG_FILTER, maxFilter);
    glObjectLabel(GL_TEXTURE, m_name, 1, "test");
}

Texture::~Texture()
{
    if (glfwGetCurrentContext() != nullptr)
    {
        glDeleteTextures(1, &m_name);
    }
    util::getGLerror(__LINE__, __FUNCTION__);
    std::cout << "texture destructor called" << std::endl;
}

void Texture::loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat, GLenum format, GLenum type, int desiredChannels)
{
    int imageWidth, imageHeight, numChannels;
    if (type != GL_FLOAT)
    {
        stbi_set_flip_vertically_on_load(true);
        const auto imageData = stbi_load(texturePath.string().c_str(), &imageWidth, &imageHeight, &numChannels, desiredChannels);

        if (!imageData)
            throw std::runtime_error("Image couldn't be loaded");

        glTextureStorage2D(m_name, 1, internalFormat, imageWidth, imageHeight);
        glTextureSubImage2D(m_name, 0, 0, 0, imageWidth, imageHeight, format, type, imageData);

        // let the cpu data of the image go
        stbi_image_free(imageData);
    }
    else
    {
        stbi_set_flip_vertically_on_load(true);
        const auto imageFloat = stbi_loadf(texturePath.string().c_str(), &imageWidth, &imageHeight, &numChannels, desiredChannels);

        if (!imageFloat)
            throw std::runtime_error("Image couldn't be loaded");

        glTextureStorage2D(m_name, 1, internalFormat, imageWidth, imageHeight);
        glTextureSubImage2D(m_name, 0, 0, 0, imageWidth, imageHeight, format, type, imageFloat);

        // let the cpu data of the image go
        stbi_image_free(imageFloat);
        stbi_set_flip_vertically_on_load(false);
    }

    m_width = imageWidth;
    m_height = imageHeight;
}

GLuint64 Texture::generateHandle()
{
    m_handle = glGetTextureHandleARB(m_name);
    if (m_handle == 0)
        throw std::runtime_error("Texture handle could not be returned");
    glMakeTextureHandleResidentARB(m_handle);
    return m_handle;
}

void Texture::initWithoutData(int width, int height, GLenum internalFormat)
{
    glTextureStorage2D(m_name, 1, internalFormat, width, height);
    m_width = width;
    m_height = height;
}

void Texture::initWithoutData3D(int width, int height, int depth, GLenum internalFormat)
{
	glTextureStorage3D(m_name, 1, internalFormat, width, height, depth);
	m_width = width;
	m_height = height;
	m_depth = depth;
}

void Texture::setWrap(const GLenum wrapS, const GLenum wrapT) const
{
    glTextureParameteri(m_name, GL_TEXTURE_WRAP_S, wrapS);
    glTextureParameteri(m_name, GL_TEXTURE_WRAP_T, wrapT);
    if (wrapS == GL_CLAMP_TO_BORDER && wrapT == GL_CLAMP_TO_BORDER)
    {
        glm::vec4 borderColor(1.0f);
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, value_ptr(borderColor));
    }
}

void Texture::setMinMagFilter(GLenum minFilter, GLenum magFilter) const
{
    glTextureParameteri(m_name, GL_TEXTURE_MIN_FILTER, minFilter);
    glTextureParameteri(m_name, GL_TEXTURE_MAG_FILTER, magFilter);
}

void Texture::generateMipmap() const
{
    glGenerateTextureMipmap(m_name);
}

GLuint64 Texture::getHandle() const
{
    if (m_handle == 0)
    {
        throw std::runtime_error("Texture handle not availabe. Did you create a handle yet?");
    }
    return m_handle;
}

GLuint Texture::getName() const
{
    return m_name;
}

int Texture::getWidth() const
{
    return m_width;
}

int Texture::getHeight() const
{
    return m_height;
}
