#pragma once

#include "Texture.h"

class Image : public Texture
{
public:
    void generateImageHandle(GLuint format, GLboolean layered = GL_FALSE, int layer = 0);
    void generateHandle() override;
};
