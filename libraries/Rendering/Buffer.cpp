#include "Buffer.h"

#include "Utils/UtilCollection.h"
#include <iostream>


Buffer::Buffer(GLenum target) : m_target(target) {
	glGenBuffers(1, &m_bufferHandle);
}


Buffer::~Buffer() {
    if (glfwGetCurrentContext() != nullptr) {
        glDeleteBuffers(1, &m_bufferHandle);
    }
    util::getGLerror(__LINE__, __FUNCTION__);
    std::cout << "buffer destructor called" << std::endl;
}

GLuint Buffer::getHandle() const {
	return m_bufferHandle;
}

void Buffer::bind() const {
	glBindBuffer(m_target, m_bufferHandle);
}

void Buffer::bindBase(unsigned int binding) const {
	glBindBufferBase(m_target, binding, m_bufferHandle);
}

void Buffer::unmapBuffer() const {
    glUnmapBuffer(m_target);

}