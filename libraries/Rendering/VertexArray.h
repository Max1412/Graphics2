#pragma once

#include "Buffer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class VertexArray {
public:
	VertexArray();

	void bind();

	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized);
	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, const GLvoid* pointer);
private:
	GLuint m_vaoHandle;
};