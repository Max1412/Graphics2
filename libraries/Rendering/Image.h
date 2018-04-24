#pragma once

#include "Texture.h"

class Image : public Texture
{
public:
    explicit Image(GLenum target = GL_TEXTURE_2D, GLenum minFilter = GL_LINEAR, GLenum maxFilter = GL_LINEAR);
    GLuint64 generateImageHandle(GLenum format, GLboolean layered = GL_FALSE, int layer = 0);
    GLuint64 generateHandle() override;
};
