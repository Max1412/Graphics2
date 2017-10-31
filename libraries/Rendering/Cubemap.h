#include "Texture.h"

class Cubemap : public Texture
{
public:
    explicit Cubemap(GLenum minFilter = GL_LINEAR, GLenum maxFilter = GL_LINEAR);
    void loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE) const override;
};