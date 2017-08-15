#pragma once

#include <array>
#include <vector>

#include <GL/glew.h>

class Buffer
{
public:
	Buffer(GLenum target);
    ~Buffer();

    /**
	 * \brief returns the OpenGL buffer handle
	 * \return buffer handle
	 */
	GLuint getHandle() const;

    /**
	 * \brief binds the buffer to the OpenGL context
	 */
	void bind() const;

    /**
	 * \brief binds the buffer to a binding layout
	 * \param binding 
	 */
	void bindBase(unsigned int binding) const;

    /**
	 * \brief binds the buffer, uses glBufferData to allocate & move data to GPU
	 * \tparam T data type
	 * \param data input data as vector
	 * \param drawType gl draw type (GL_..._DRAW)
	 */
	template<typename T>
	void setData(const std::vector<T> &data,  GLenum drawType);

    /**
	 * \brief binds the buffer, uses glBufferData to allocate & move data to GPU
	 * \tparam T data type
	 * \tparam N array size
	 * \param data input data as array
	 * \param drawType gl draw type (GL_..._DRAW)
	 */
	template<typename T, std::size_t N>
	void setData(const std::array<T, N> &data, GLenum drawType);

    /**
	 * \brief binds the buffer, uses glBufferStorage, buffer will be immutable
	 * \tparam T data type
	 * \param data input data as vector
	 * \param flags buffer flags
	 */
	template<typename T>
	void setStorage(const std::vector<T> &data, GLbitfield flags);

    /**
	 * \brief binds the buffer, uses glBufferStorage, buffer will be immutable
	 * \tparam T data type
	 * \tparam N array size
	 * \param data input data as vector
	 * \param flags buffer flags
	 */
	template<typename T, std::size_t N>
	void setStorage(const std::array<T, N> &data, GLbitfield flags);

    /**
	 * \brief binds the buffer, sets (sub)data
	 * \tparam T data type
	 * \param data input data as array
	 * \param offset offset in bytes from the start
	 */
	template<typename T>
	void setSubData(const std::vector<T> &data, int offset);

    /**
	 * \brief binds the buffer, sets (sub)data
	 * \tparam T data type
	 * \tparam N array size
	 * \param data input data as array
	 * \param offset offset in bytes from the start
	 */
	template<typename T, std::size_t N>
	void setSubData(const std::array<T, N> &data, int offset);

    /**
     * \brief maps the buffer, writes data, unmaps the buffer
     * \tparam S data type
     * \param data input to write into the buffer
     * \param startOffset offset in the buffer in bytes to write the data
     */
    template<typename S>
    void setPartialContentMapped(const S& data, int startOffset);

    /**
     * \brief maps the buffer, returns mapped pointer
     * \tparam S content data type
     * \param size size of mapped part of the buffer
     * \param startOffset start offset to the buffer data
     * \param flags buffer flags
     * \return 
     */
    template <class S>
    S* mapBufferContent(int size, int startOffset, GLbitfield flags);

    /**
     * \brief unmaps the buffer
     */
    void unmapBuffer() const;

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
S* Buffer::mapBufferContent(int size, int startOffset, GLbitfield flags) {
    bind();
    return static_cast<S*>(glMapBufferRange(m_target, startOffset, size, flags));
}

template<typename S>
void Buffer::setContentSubData(const S& data, int startOffset) {
    bind();
    glBufferSubData(m_target, startOffset, sizeof(data), &data);
}

template <typename T>
void Buffer::setData(const std::vector<T> &data, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	glBindBuffer(m_target, m_bufferHandle);
	glBufferData(m_target, data.size() * sizeof(T), data.data(), drawType);
}

template <typename T, std::size_t N>
void Buffer::setData(const std::array<T, N> &data, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	glBindBuffer(m_target, m_bufferHandle);
	glBufferData(m_target, data.size() * sizeof(T), data.data(), drawType);
}

template<typename T>
void Buffer::setSubData(const std::vector<T> &data, int offset) {
	glBindBuffer(m_target, m_bufferHandle);
	glBufferSubData(m_target, offset, data.size() * sizeof(T), data.data());
}

template <typename T, std::size_t N>
void Buffer::setSubData(const std::array<T, N> &data, int offset) {
	glBindBuffer(m_target, m_bufferHandle);
	glBufferSubData(m_target, offset, data.size() * sizeof(T), data.data());
}

template<typename T>
void Buffer::setStorage(const std::vector<T> &data, GLbitfield flags) {
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	glBindBuffer(m_target, m_bufferHandle);
	glBufferStorage(m_target, data.size() * sizeof(T), data.data(), flags);
	m_isImmutable = true;
}

template <typename T, std::size_t N>
void Buffer::setStorage(const std::array<T, N> &data, GLbitfield flags) {
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	glBindBuffer(m_target, m_bufferHandle);
	glBufferStorage(m_target, data.size() * sizeof(T), data.data(), flags);
	m_isImmutable = true;

}