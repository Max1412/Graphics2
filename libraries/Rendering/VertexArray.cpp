#include "VertexArray.h"
#include "Utils/UtilCollection.h"

VertexArray::VertexArray() {
	glGenVertexArrays(1, &m_vaoHandle);
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

void VertexArray::connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized) const {
	bind();
	glEnableVertexAttribArray(index);
	buffer.bind();
	glVertexAttribPointer(index, size, type, normalized, 0, (GLubyte *)nullptr);
}

void VertexArray::connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, const GLvoid* pointer) const {
	bind();
	glEnableVertexAttribArray(index);
	buffer.bind();
	glVertexAttribPointer(index, size, type, normalized, stride, pointer);
}