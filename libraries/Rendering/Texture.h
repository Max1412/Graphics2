#pragma once
#include <GL/glew.h>
#include <filesystem>

class Texture
{
public:
    explicit Texture(GLenum target = GL_TEXTURE_2D, GLenum minFilter = GL_LINEAR, GLenum maxFilter = GL_LINEAR);

    void loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE) const;

    virtual void generateHandle();

    void initWithoutData(int width, int height, GLenum internalFormat) const;

    template <typename T>
    void clearTexture(GLenum format, GLenum type, T data) const
    {
        glClearTexImage(m_name, 0, format, type, &data);
    }

    GLuint64 getHandle() const;

protected:
    GLuint m_name;
    GLuint64 m_handle = 0;
};

