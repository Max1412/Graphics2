#pragma once

#include <array>
#include <vector>

#include <GL/glew.h>

class Buffer
{
public:
	explicit Buffer(GLenum target);
    ~Buffer();

    /**
	 * \brief returns the OpenGL buffer handle
	 * \return buffer handle
	 */
	GLuint getHandle() const;

    /**
    * \brief returns the OpenGL buffer target (GL_*_BUFFER)
    * \return buffer handle
    */
    GLuint getTarget() const;

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
     * \brief maps the buffer, writes data, unmaps the buffer
     * \tparam S data type
     * \param data input to write into the buffer
     * \param startOffset offset in the buffer in bytes to write the data
     */
    template<typename S>
    void setPartialContentMapped(const S& data, unsigned startOffset);

    /**
     * \brief maps the buffer, returns mapped pointer
     * \tparam S content data type
     * \param size size of mapped part of the buffer
     * \param startOffset start offset to the buffer data
     * \param flags buffer flags
     * \return 
     */
    template <class S>
    S* mapBufferContent(int size, unsigned startOffset, GLbitfield flags);

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
    void setContentSubData(const S& data, unsigned startOffset);

private:
	GLuint m_bufferHandle;
	GLenum m_target;

	bool m_isImmutable = false;
};

/*
*  Initializing template functions
*/

//
// BufferData Functions (mutable)
// if called a second time on the same buffer, memory will get newly allocated
//
template <typename T>
void Buffer::setData(const std::vector<T> &data, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	glNamedBufferData(m_bufferHandle, data.size() * sizeof(T), data.data(), drawType);
}

template <typename T, std::size_t N>
void Buffer::setData(const std::array<T, N> &data, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
	glNamedBufferData(m_bufferHandle, data.size() * sizeof(T), data.data(), drawType);
}

//
// BufferStorage Functions (immutable)
// call these only once on the same buffer
//
template<typename T>
void Buffer::setStorage(const std::vector<T> &data, GLbitfield flags) {
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
    glNamedBufferStorage(m_bufferHandle, data.size() * sizeof(T), data.data(), flags);
    m_isImmutable = true;
}

template <typename T, std::size_t N>
void Buffer::setStorage(const std::array<T, N> &data, GLbitfield flags) {
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
    glNamedBufferStorage(m_bufferHandle, data.size() * sizeof(T), data.data(), flags);
    m_isImmutable = true;
}

/*
*  Non-Initializing template functions
*/

// set the buffer data by using SubData
template<typename S>
void Buffer::setContentSubData(const S& data, unsigned startOffset) {
    glNamedBufferSubData(m_bufferHandle, startOffset, sizeof(data), &data);
}

// return a mapped pointer in order to set data in the buffer
template<typename S>
S* Buffer::mapBufferContent(int size, unsigned startOffset, GLbitfield flags) {
    return static_cast<S*>(glMapNamedBufferRange(m_bufferHandle, startOffset, size, flags));
}

// set the buffer data by mapping
template<typename S>
void Buffer::setPartialContentMapped(const S& data, unsigned startOffset) {
    S* ptr = static_cast<S*>(glMapNamedBufferRange(m_bufferHandle, startOffset, sizeof(data), GL_MAP_WRITE_BIT));
    *ptr = data;
    unmapBuffer();
}
