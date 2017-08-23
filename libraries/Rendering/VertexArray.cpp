#include "VertexArray.h"
#include "Utils/UtilCollection.h"

VertexArray::VertexArray() {
	glCreateVertexArrays(1, &m_vaoHandle);
}

VertexArray::~VertexArray() {
    if (glfwGetCurrentContext() != nullptr) {
        glDeleteVertexArrays(1, &m_vaoHandle);
    }
	util::getGLerror(__LINE__, __FUNCTION__);
}

void VertexArray::bind() const {
	glBindVertexArray(m_vaoHandle);
}

void VertexArray::connectIndexBuffer(Buffer &buffer) const {
    if (buffer.getTarget() != GL_ELEMENT_ARRAY_BUFFER)
        throw std::runtime_error("Only index buffers can be conntected using connectIndexBuffer");
    glVertexArrayElementBuffer(m_vaoHandle, buffer.getHandle());
}


void VertexArray::connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized) const {
    glEnableVertexArrayAttrib(m_vaoHandle, index);
    glVertexArrayVertexBuffer(m_vaoHandle, index, buffer.getHandle(), 0, 4 * size); // only works for 32 bit types for now
    glVertexArrayAttribFormat(m_vaoHandle, index, size, type, normalized, 0);
    glVertexArrayAttribBinding(m_vaoHandle, index, index);
}


void VertexArray::connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, GLuint offset, GLuint relativeOffset) const {
    glEnableVertexArrayAttrib(m_vaoHandle, index);
    glVertexArrayVertexBuffer(m_vaoHandle, index, buffer.getHandle(), offset, stride);
    glVertexArrayAttribFormat(m_vaoHandle, index, size, type, normalized, relativeOffset);
    glVertexArrayAttribBinding(m_vaoHandle, index, index);
}