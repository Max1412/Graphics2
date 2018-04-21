#pragma once
#include <glbinding/gl/gl.h>
using namespace gl;
#include <filesystem>

class Texture
{
public:
    explicit Texture(GLenum target = GL_TEXTURE_2D, GLenum minFilter = GL_LINEAR, GLenum maxFilter = GL_LINEAR);

    virtual void loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int desiredChannels = 4);
    virtual void generateHandle();

    virtual void initWithoutData(int width, int height, GLenum internalFormat);

    void setWrap(GLenum wrapS, GLenum wrapT) const;
    void setMinMagFilter(GLenum minFilter, GLenum magFilter) const;
    void generateMipmap() const;

    template <typename T>
    void clearTexture(GLenum format, GLenum type, T data) const
    {
        glClearTexImage(m_name, 0, format, type, &data);
    }

    GLuint64 getHandle() const;
    GLuint getName() const;

    int getWidth() const;
    int getHeight() const;

protected:
    GLuint m_name;
    GLuint64 m_handle = 0;
    int m_width = 0;
    int m_height = 0;
};
