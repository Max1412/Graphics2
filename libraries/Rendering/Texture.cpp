#include "Texture.h"
#include "IO/stb_image.h"


Texture::Texture(GLenum target, GLenum minFilter, GLenum maxFilter)
{
    glCreateTextures(target, 1, &m_name);
    glTextureParameteri(m_name, GL_TEXTURE_MIN_FILTER, minFilter);
    glTextureParameteri(m_name, GL_TEXTURE_MAG_FILTER, maxFilter);
}

void Texture::loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat, GLenum format, GLenum type) const
{
    auto imageFilename(RESOURCES_PATH + std::string("/image.png"));
    int imageWidth, imageHeight, numChannels;
    const auto imageData = stbi_load(imageFilename.c_str(), &imageWidth, &imageHeight, &numChannels, 4);

    if (!imageData)
        throw std::runtime_error("Image couldn't be loaded");

    glTextureStorage2D(m_name, 1, internalFormat, imageWidth, imageHeight);
    glTextureSubImage2D(m_name, 0, 0, 0, imageWidth, imageHeight, format, type, imageData);

    // let the cpu data of the image go
    stbi_image_free(imageData);
}

void Texture::generateHandle()
{
    m_handle = glGetTextureHandleARB(m_name);
    if (m_handle == 0)
        throw std::runtime_error("Texture handle could not be returned");
    glMakeTextureHandleResidentARB(m_handle);
}

void Texture::initWithoutData(int width, int height, GLenum internalFormat) const
{
    glTextureStorage2D(m_name, 1, internalFormat, width, height);
}

GLuint64 Texture::getHandle() const
{
    if(m_handle == 0)
    {
        throw std::runtime_error("Texture handle not availabe. Did you create a handle yet?");
    }
    return m_handle;
}
