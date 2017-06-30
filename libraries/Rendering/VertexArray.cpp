#include "VertexArray.h"

VertexArray::VertexArray() {
	glGenVertexArrays(1, &m_vaoHandle);
}

void VertexArray::bind() {
	glBindVertexArray(m_vaoHandle);
}

void VertexArray::connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized) {
	bind();
	glEnableVertexAttribArray(index);
	buffer.bind();
	glVertexAttribPointer(index, size, type, normalized, 0, (GLubyte *)NULL);
}

void VertexArray::connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, const GLvoid* pointer) {
	bind();
	glEnableVertexAttribArray(index);
	buffer.bind();
	glVertexAttribPointer(index, size, type, normalized, stride, pointer);
}