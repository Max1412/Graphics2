#pragma once

#include <array>
#include <vector>

#include <glbinding/gl/gl.h>
using namespace gl;
#include "Utils/UtilCollection.h"

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
    GLenum getTarget() const;

    /**
     * \brief returns the size of the type of the data elemts in the buffer (e.g. sizeof(glm::vec3))
     * \return size of underlying data type
     */
    size_t getTypeSize() const;

    /**
     * \brief binds the buffer to a binding layout
     * \param binding 
     */
    void bindBase(unsigned int binding) const;

    /**
     * \brief uses glBufferStorage, buffer will be immutable
     * \tparam T data type
     * \param data input data as vector
     * \param flags buffer flags
     */
    template <typename T>
    void setStorage(const std::vector<T>& data, BufferStorageMask flags);

    /**
     * \brief uses glBufferStorage, buffer will be immutable
     * \tparam T data type
     * \tparam N array size
     * \param data input data as array
     * \param flags buffer flags
     */
    template <typename T, size_t N>
    void setStorage(const std::array<T, N>& data, BufferStorageMask flags);

    /**
     * \brief maps the buffer, writes data, unmaps the buffer
     * \tparam S data type
     * \param data input to write into the buffer
     * \param startOffset offset in the buffer in bytes to write the data
     */
    template <typename S>
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
    S* mapBufferContent(size_t size, size_t startOffset, BufferAccessMask flags);

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
    template <typename S>
    void setContentSubData(const S& data, size_t startOffset);

private:
    GLuint m_bufferHandle;
    GLenum m_target;

    size_t m_typeSize = 0;

    bool m_isImmutable = false;

    BufferStorageMask m_bufferFlags = GL_NONE_BIT;
};

/*
*  Initializing template functions
*/

//
// BufferStorage Functions (immutable)
// call these only once on the same buffer
//
template <typename T>
void Buffer::setStorage(const std::vector<T>& data, BufferStorageMask flags)
{
    util::getGLerror(__LINE__, __FUNCTION__);
    m_bufferFlags = flags;
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
    glNamedBufferStorage(m_bufferHandle, data.size() * sizeof(T), data.data(), flags);
    m_isImmutable = true;
    m_typeSize = sizeof(T);
    util::getGLerror(__LINE__, __FUNCTION__);
}

template <typename T, size_t N>
void Buffer::setStorage(const std::array<T, N>& data, BufferStorageMask flags)
{
    util::getGLerror(__LINE__, __FUNCTION__);
    m_bufferFlags = flags;
    if (m_isImmutable)
        throw std::runtime_error("Buffer is immutable, cannot reallocate buffer data");
    glNamedBufferStorage(m_bufferHandle, data.size() * sizeof(T), data.data(), flags);
    m_isImmutable = true;
    m_typeSize = sizeof(T);
    util::getGLerror(__LINE__, __FUNCTION__);
}

/*
*  Non-Initializing template functions
*/

// set the buffer data by using SubData
template <typename S>
void Buffer::setContentSubData(const S& data, size_t startOffset)
{
    // TODO turn thse into asserts, only check them in debug mode
    if constexpr(util::debugmode)
    {
        if ((m_bufferFlags & BufferStorageMask::GL_DYNAMIC_STORAGE_BIT) == BufferStorageMask::GL_NONE_BIT)
        {
            throw std::runtime_error("Buffer lacks GL_DYNAMIC_STORAGE_BIT flag for using SubData");
        }
    }
    glNamedBufferSubData(m_bufferHandle, startOffset, sizeof(data), &data);
}

// return a mapped pointer in order to set data in the buffer
template <typename S>
S* Buffer::mapBufferContent(size_t size, size_t startOffset, BufferAccessMask flags)
{
    // TODO turn thse into asserts, only check them in debug mode
    // TODO what about COHERENT_BIT, PERSISTENT_BIT
    if constexpr(util::debugmode)
    {
        if ((m_bufferFlags & BufferStorageMask::GL_MAP_WRITE_BIT) == BufferStorageMask::GL_NONE_BIT 
            && (m_bufferFlags & BufferStorageMask::GL_MAP_READ_BIT) == BufferStorageMask::GL_NONE_BIT)
        {
            throw std::runtime_error("Buffer needs GL_MAP_WRITE_BIT or GL_MAP_READ_BIT to use mapping");
        }
        if (!((static_cast<BufferAccessMask>(m_bufferFlags) & flags) == flags))
        {
            throw std::runtime_error("Buffer needs to have the flags set that mapBuffer gets called with");
        }
    }
    S* ptr = static_cast<S*>(glMapNamedBufferRange(m_bufferHandle, startOffset, size, flags));
    util::getGLerror(__LINE__, __FUNCTION__);

    if constexpr(util::debugmode)
    {
        if (!ptr)
        {
            throw std::runtime_error("Mapping the buffer failed");
        }
    }

    return ptr;
}

// set the buffer data by mapping
template <typename S>
void Buffer::setPartialContentMapped(const S& data, unsigned startOffset)
{
    // TODO turn thse into asserts, only check them in debug mode
    // TODO what about COHERENT_BIT, PERSISTENT_BIT
    if constexpr(util::debugmode)
    {
        if ((m_bufferFlags & BufferStorageMask::GL_MAP_WRITE_BIT) == BufferStorageMask::GL_NONE_BIT)
        {
            throw std::runtime_error("Buffer lacks GL_MAP_WRITE_BIT to write through mapping");
        }
    }
    S* ptr = static_cast<S*>(glMapNamedBufferRange(m_bufferHandle, startOffset, sizeof(data), GL_MAP_WRITE_BIT));
    if constexpr(util::debugmode)
    {
        if (!ptr)
        {
            throw std::runtime_error("Mapping the buffer failed");
        }
    }
    *ptr = data;
    unmapBuffer();
}
