#include "Image.h"

void Image::generateImageHandle(GLuint format, GLboolean layered, int layer)
{
	m_handle = glGetImageHandleARB(m_name, 0, layered, layer, format);
	if (m_handle == 0)
		throw std::runtime_error("iamge handle could not be returned");
	glMakeImageHandleResidentARB(m_handle, GL_READ_WRITE);
}

void Image::generateHandle()
{
	throw std::runtime_error("generateHandle() cannot be used on an Image. Use generateImageHanlde() instead");
}
