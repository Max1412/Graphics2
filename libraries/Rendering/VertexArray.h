#pragma once

#include "Buffer.h"

class VertexArray {
public:
	VertexArray();

	void bind() const;

	void del();

	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized);
	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, const GLvoid* pointer);
private:
	GLuint m_vaoHandle;
};