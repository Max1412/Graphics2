#include "Cubemap.h"
#include "stb/stb_image.h"
#include <sstream>

Cubemap::Cubemap(GLenum minFilter, GLenum maxFilter) : Texture(GL_TEXTURE_CUBE_MAP, minFilter, maxFilter)
{
    
}


void Cubemap::initWithoutData(int width, int height, GLenum internalFormat, GLenum format, GLenum type, const int levels)
{
    glTextureStorage2D(m_name, levels, internalFormat, width, height);
    m_width = width;
    m_height = height;

    glTextureParameteri(m_name, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_name, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_name, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_name, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_name, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //for (int face = 0; face < 6; face++)
    //{
    //    glTextureSubImage3D(
    //        m_name,
    //        0,      // only 1 level in example
    //        0,
    //        0,
    //        face,   // the offset to desired cubemap face, which offset goes to which face above
    //        width,
    //        height,
    //        1,      // depth how many faces to set, if this was 3 we'd set 3 cubemap faces at once
    //        format,
    //        type,
    //        nullptr);
    //}
}

void Cubemap::loadFromFile(const std::experimental::filesystem::path& texturePath, GLenum internalFormat, GLenum format, GLenum type, int desiredChannels)
{
    // load the first image to get the width, height
    std::string path(texturePath.string());
    path.insert(path.cbegin() + path.find_last_of('.'), 1, '0'); // image.png -> image1.png

    int imageWidth, imageHeight, numChannels;
    auto imageData = stbi_load(path.c_str(), &imageWidth, &imageHeight, &numChannels, desiredChannels);

    if (!imageData)
        throw std::runtime_error("Cubemap Image couldn't be loaded");

    m_width = imageWidth;
    m_height = imageHeight;

    glTextureStorage2D(m_name, 1, internalFormat, imageWidth, imageHeight);

    glTextureParameteri(m_name, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_name, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_name, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // load the data for all faces
    for (int face = 0; face < 6; face++)
    {
        // face:
        // 0 = positive x face
        // 1 = negative x face
        // 2 = positive y face
        // 3 = negative y face
        // 4 = positive z face
        // 5 = negative z face

        path = texturePath.string();
        path.insert(path.cbegin() + path.find_last_of('.'), 1, std::to_string(face).c_str()[0]); // image.png -> image1.png, image2.png, ...

        imageData = stbi_load(path.c_str(), &imageWidth, &imageHeight, &numChannels, 4);

        if (!imageData)
        {
            std::stringstream ss;
            ss << "Cubemap Image " << face << " couldn't be loaded";
            throw std::runtime_error(ss.str());
        }

        glTextureSubImage3D(
            m_name,
            0,      // only 1 level in example
            0,
            0,
            face,   // the offset to desired cubemap face, which offset goes to which face above
            imageWidth,
            imageHeight,
            1,      // depth how many faces to set, if this was 3 we'd set 3 cubemap faces at once
            format,
            type,
            imageData);

        // let the cpu data of the image go
        stbi_image_free(imageData);
    }
}

