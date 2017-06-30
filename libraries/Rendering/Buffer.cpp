#include "Buffer.h"

Buffer::Buffer() {
	glGenBuffers(1, &m_bufferHandle);
}

GLuint Buffer::getHandle() {
	return m_bufferHandle;
}

void Buffer::bind() {
	glBindBuffer(m_target, m_bufferHandle);
}