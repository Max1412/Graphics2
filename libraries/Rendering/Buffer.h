#pragma once

#include <array>
#include <vector>

#include <GL/glew.h>

class Buffer
{
public:
	Buffer();

    /**
	 * \brief returns the OpenGL buffer handle
	 * \return buffer handle
	 */
	GLuint getHandle();

    /**
	 * \brief binds the buffer to the OpenGL context
	 */
	void bind() const;

    /**
	 * \brief binds the buffer to a binding layout
	 * \param binding 
	 */
	void bindBase(unsigned int binding);

    /**
	 * \brief current destructor workaround
	 */
	void del();

    /**
	 * \brief binds the buffer, uses glBufferData to allocate & move data to GPU
	 * \tparam T data type
	 * \param data input data as vector
	 * \param target gl buffer target (GL_..._BUFFER)
	 * \param drawType gl draw type (GL_..._DRAW)
	 */
	template<typename T>
	void setData(const std::vector<T> &data, GLenum target,  GLenum drawType);

    /**
	 * \brief binds the buffer, uses glBufferData to allocate & move data to GPU
	 * \tparam T data type
	 * \tparam N array size
	 * \param data input data as array
	 * \param target gl buffer target (GL_..._BUFFER)
	 * \param drawType gl draw type (GL_..._DRAW)
	 */
	template<typename T, std::size_t N>
	void setData(const std::array<T, N> &data, GLenum target, GLenum drawType);

    /**
	 * \brief binds the buffer, uses glBufferStorage, buffer will be immutable
	 * \tparam T data type
	 * \param data input data as vector
	 * \param target gl buffer target (GL_..._BUFFER)
	 * \param flags buffer flags
	 */
	template<typename T>
	void setStorage(const std::vector<T> &data, GLenum target, GLbitfield flags);

    /**
	 * \brief binds the buffer, uses glBufferStorage, buffer will be immutable
	 * \tparam T data type
	 * \tparam N array size
	 * \param data input data as vector
	 * \param target gl buffer target (GL_..._BUFFER)
	 * \param flags buffer flags
	 */
	template<typename T, std::size_t N>
	void setStorage(const std::array<T, N> &data, GLenum target, GLbitfield flags);

    /**
	 * \brief binds the buffer, sets (sub)data
	 * \tparam T data type
	 * \param data input data as array
	 * \param target gl buffer target (GL_..._BUFFER)
	 * \param offset offset in bytes from the start
	 */
	template<typename T>
	void setSubData(const std::vector<T> &data, GLenum target, int offset);

    /**
	 * \brief binds the buffer, sets (sub)data
	 * \tparam T data type
	 * \tparam N array size
	 * \param data input data as array
	 * \param target gl buffer target (GL_..._BUFFER)
	 * \param offset offset in bytes from the start
	 */
	template<typename T, std::size_t N>
	void setSubData(const std::array<T, N> &data, GLenum target, int offset);

    /**
     * \brief maps the buffer, writes data, unmaps the buffer
     * \tparam S data type
     * \param data input to write into the buffer
     * \param startOffset offset in the buffer in bytes to write the data
     */
    template<typename S>
    void setPartialContentMapped(const S& data, int startOffset);
    template <class S>
    S* mapBufferContet(int size, int startOffset, GLbitfield flags);
    void unmapBuffer();

    /**
     * \brief binds the buffer, writes data with glBufferSubData
     * \tparam S data type
     * \param data input to write into the buffer
     * \param startOffset offset in the buffer in bytes to write the data
     */
    template<typename S>
    void setContentSubData(const S& data, int startOffset);

private:
	GLuint m_bufferHandle;
	GLenum m_target;

	bool m_isImmutable = false;


};

/*
*  Template function must be implemented in the header (whyyyy)
*/

template<typename S>
void Buffer::setPartialContentMapped(const S& data, int startOffset) {
    bind();
    S* ptr = static_cast<S*>(glMapBufferRange(m_target, startOffset, sizeof(data), GL_MAP_WRITE_BIT));
    *ptr = data;
    glUnmapBuffer(m_target);
}

template<typename S>
S* Buffer::mapBufferContet(int size, int startOffset, GLbitfield flags) {
    bind();
    return static_cast<S*>(glMapBufferRange(m_target, startOffset, size, flags));
}

template<typename S>
void Buffer::setContentSubData(const S& data, int startOffset) {
    bind();
    glBufferSubData(m_target, startOffset, sizeof(data), &data);
}

template <typename T>
void Buffer::setData(const std::vector<T> &data, GLenum target, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferData(target, data.size() * sizeof(T), data.data(), drawType);
}

template <typename T, std::size_t N>
void Buffer::setData(const std::array<T, N> &data, GLenum target, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferData(target, data.size() * sizeof(T), data.data(), drawType);
}

template<typename T>
void Buffer::setSubData(const std::vector<T> &data, GLenum target, int offset) {
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferSubData(target, offset, data.size() * sizeof(T), data.data());
}

template <typename T, std::size_t N>
void Buffer::setSubData(const std::array<T, N> &data, GLenum target, int offset) {
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferSubData(target, offset, data.size() * sizeof(T), data.data());
}

template<typename T>
void Buffer::setStorage(const std::vector<T> &data, GLenum target, GLbitfield flags) {
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferStorage(target, data.size() * sizeof(T), data.data(), flags);
	m_isImmutable = true;
}

template <typename T, std::size_t N>
void Buffer::setStorage(const std::array<T, N> &data, GLenum target, GLbitfield flags) {
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferStorage(target, data.size() * sizeof(T), data.data(), flags);
	m_isImmutable = true;

}