#pragma once

#include "Buffer.h"

class VertexArray {
public:
	VertexArray();
    ~VertexArray();

    /**
	 * \brief binds the VAO
	 */
	void bind() const;

    /**
	 * \brief connects a buffer to this vao, assumes stride/pointer = 0
	 * \param buffer buffer to connect to this vao
	 * \param index location to be used
	 * \param size elements per buffer primitive
	 * \param type GL data type
	 * \param normalized specifies if fixed point attributes should be normalized
	 */
	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized) const;
   
    /**
	 * \brief connects a buffer to this vao
	 * \param buffer buffer to connect to this vao
	 * \param index location to be used
	 * \param size elements per buffer primitive
	 * \param type GL data type
	 * \param normalized specifies if fixed point attributes should be normalized
	 * \param stride byte offset between vertex attributes
	 * \param pointer offset to begin of the array
	 */
	void connectBuffer(Buffer &buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, const GLvoid* pointer) const;
private:
	GLuint m_vaoHandle;
};