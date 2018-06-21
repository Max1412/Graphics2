#pragma once

#include "Texture.h"

class Cubemap : public Texture
{
public:
    explicit Cubemap(GLenum minFilter = GL_LINEAR, GLenum maxFilter = GL_LINEAR);
    TextureLoadInfo loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, int desiredChannels = 4) override;
    void initWithoutData(int width, int height, GLenum internalFormat, GLenum format, GLenum type, int levels = 1);
};
