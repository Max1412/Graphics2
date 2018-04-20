#pragma once

#include "Buffer.h"

class VertexArray
{
public:
	VertexArray();
	~VertexArray();

	/**
	 * \brief connects an index buffer (GL_ELEMENT_ARRAY_BUFFER) to the vao
	 * \param buffer the index buffer to be connected
	 */
	void connectIndexBuffer(Buffer& buffer) const;

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
	 * \warning only works for 32 bit (because of stride) floating point (because of missing I or L) types for now. 
	 */
	void connectBuffer(const Buffer& buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized) const;

	/**
	 * \brief connects a buffer to this vao
	 * \param buffer buffer to connect to this vao
	 * \param index location to be used
	 * \param size elements per buffer primitive
	 * \param type GL data type
	 * \param normalized specifies if fixed point attributes should be normalized
	 * \param stride byte offset between vertex attributes
	 * \param offset offset of the first element of the buffer
	 * \param relativeOffset distance between elements within the buffer
	 */
	void connectBuffer(const Buffer& buffer, GLuint index, GLuint size, GLenum type, GLboolean normalized, GLuint stride, GLuint offset, GLuint relativeOffset) const;
private:
	GLuint m_vaoHandle;
};
