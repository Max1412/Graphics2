#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <array>
#include <vector>

class Buffer
{
public:
	Buffer();

	GLuint getHandle();

	void bind();

	template<typename T>
	void setData(std::vector<T> &data, GLenum target,  GLenum drawType);

	template<typename T, std::size_t N>
	void setData(std::array<T, N> &data, GLenum target, GLenum drawType);

	template<typename T>
	void setStorage(std::vector<T> &data, GLenum target, GLbitfield flags);

	template<typename T, std::size_t N>
	void setStorage(std::array<T, N> &data, GLenum target, GLbitfield flags);

	template<typename T>
	void setSubData(std::vector<T> &data, GLenum target, int offset);

	template<typename T, std::size_t N>
	void setSubData(std::array<T, N> &data, GLenum target, int offset);

private:
	GLuint m_bufferHandle;
	GLenum m_target;

	bool m_isImmutable = false;


};

/*
*  Template function must be implemented in the header (whyyyy)
*/

template <typename T>
void Buffer::setData(std::vector<T> &data, GLenum target, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reassign buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferData(target, data.size() * sizeof(T), data.data(), drawType);
}

template <typename T, std::size_t N>
void Buffer::setData(std::array<T, N> &data, GLenum target, GLenum drawType) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reassign buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferData(target, data.size() * sizeof(T), data.data(), drawType);
}

template<typename T>
void Buffer::setSubData(std::vector<T> &data, GLenum target, int offset) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reassign buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferSubData(target, offset, data.size() * sizeof(T), data.data());
}

template <typename T, std::size_t N>
void Buffer::setSubData(std::array<T, N> &data, GLenum target, int offset) {
	if (m_isImmutable)
		throw std::runtime_error("Buffer is immutable, cannot reassign buffer data");
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferSubData(target, offset, data.size() * sizeof(T), data.data());
}

template<typename T>
void Buffer::setStorage(std::vector<T> &data, GLenum target, GLbitfield flags) {
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferStorage(target, data.size() * sizeof(T), data.data(), flags);
	m_isImmutable = true;

}

template <typename T, std::size_t N>
void Buffer::setStorage(std::array<T, N> &data, GLenum target, GLbitfield flags) {
	m_target = target;
	glBindBuffer(target, m_bufferHandle);
	glBufferStorage(target, data.size() * sizeof(T), data.data(), flags);
	m_isImmutable = true;

}