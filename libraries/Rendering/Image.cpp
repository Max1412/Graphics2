#include "Image.h"
#include <iostream>

Image::Image(GLenum target, GLenum minFilter, GLenum maxFilter) : Texture(target, minFilter, maxFilter)
{

}

GLuint64 Image::generateImageHandle(GLenum format, GLboolean layered, int layer)
{
    m_handle = glGetImageHandleARB(m_name, 0, layered, layer, format);
    if (m_handle == 0)
        throw std::runtime_error("image handle could not be returned");
    glMakeImageHandleResidentARB(m_handle, GL_READ_WRITE);
    return m_handle;
}

GLuint64 Image::generateHandle()
{
    std::cout << "WARNING: Generating texture handle for image! Consider calling generateImageHandle() instead. \n";
    return Texture::generateHandle();
}
