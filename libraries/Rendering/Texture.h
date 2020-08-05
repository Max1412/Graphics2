#pragma once
#include <glbinding/gl/gl.h>
using namespace gl;
#include <filesystem>
#include <glm/glm.hpp>

enum class TextureLoadInfo
{
    None,
    Transparent,
    Opaque,
    Other
};

class Texture
{
public:
    explicit Texture(GLenum target = GL_TEXTURE_2D, GLenum minFilter = GL_LINEAR_MIPMAP_LINEAR, GLenum maxFilter = GL_LINEAR);
    virtual ~Texture();
    virtual TextureLoadInfo loadFromFile(const std::filesystem::path& texturePath, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int desiredChannels = 4);
    virtual GLuint64 generateHandle();

	template<typename T, typename = decltype(std::data(std::declval<T>()))>
	void initWithData1D(const T& container, GLint width, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE);

	template<typename T, typename = decltype(std::data(std::declval<T>()))>
	void initWithData2D(const T& container, GLint width, GLint height, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE);

	template<typename T, typename = decltype(std::data(std::declval<T>()))>
	void initWithData3D(const T& container, GLint width, GLint height, GLint depth, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE);
	
	virtual void initWithoutData(int width, int height, GLenum internalFormat);
    virtual void initWithoutDataMultiSample(int width, int height, GLenum internalFormat, GLsizei samples, GLboolean fixedSampleLocations);
    virtual void initWithoutData3D(int width, int height, int depth, GLenum internalFormat);

    void setWrap(GLenum wrapS, GLenum wrapT) const;
    void setWrap(GLenum wrapS, GLenum wrapT, GLenum wrapR) const;
    void setMinMagFilter(GLenum minFilter, GLenum magFilter) const;
    void generateMipmap() const;

    template <typename T>
    void clearTexture(GLenum format, GLenum type, T data, GLint level = 0) const
    {
        glClearTexImage(m_name, level, format, type, &data);
    }

    TextureLoadInfo getTextureLoadInfo() const;

    GLuint64 getHandle() const;
    GLuint getName() const;

    int getWidth() const;
    int getHeight() const;

protected:
    GLuint m_name;
    GLuint64 m_handle = 0;
    int m_width = 0;
    int m_height = 0;
    int m_depth = 0;

    TextureLoadInfo m_textureLoadInfo = TextureLoadInfo::None;
};

template<typename T, typename>
void Texture::initWithData1D(const T& container, GLint width, GLenum internalFormat, GLenum format, GLenum type)
{
    glEnable(GL_TEXTURE_1D);
    const auto numLevels = static_cast<GLsizei>(glm::ceil(glm::log2<float>(static_cast<float>(width))));
	glTextureStorage1D(m_name, numLevels, internalFormat, width);
	glTextureSubImage1D(m_name, 0, 0, width, format, type, container.data());
    generateMipmap();
	m_width = width;
}

template<typename T, typename>
void Texture::initWithData2D(const T& container, GLint width, GLint height, GLenum internalFormat, GLenum format, GLenum type)
{
    glEnable(GL_TEXTURE_2D);
    const auto numLevels = static_cast<GLsizei>(glm::ceil(glm::log2<float>(static_cast<float>(glm::max(width, height)))));
	glTextureStorage2D(m_name, numLevels, internalFormat, width, height);
	glTextureSubImage2D(m_name, 0, 0, 0, width, height, format, type, container.data());
    generateMipmap();
	m_width = width;
	m_height = height;
}

template<typename T, typename>
void Texture::initWithData3D(const T& container, GLint width, GLint height, GLint depth, GLenum internalFormat, GLenum format, GLenum type)
{
    glEnable(GL_TEXTURE_3D);
    const auto numLevels = static_cast<GLsizei>(glm::ceil(glm::log2<float>(glm::max(glm::max(width, height), depth))));
	glTextureStorage3D(m_name, numLevels, internalFormat, width, height, depth);
	glTextureSubImage3D(m_name, 0, 0, 0, 0, width, height, depth, format, type, container.data());
    generateMipmap();
	m_width = width;
	m_height = height;
	m_depth = depth;
}
