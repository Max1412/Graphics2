#pragma once

#include "Buffer.h"

#include "Utils/UtilCollection.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class VertexArray {
public:
	VertexArray();

	void bind();

	void del();

	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized);
	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, const GLvoid* pointer);
private:
	GLuint m_vaoHandle;
};