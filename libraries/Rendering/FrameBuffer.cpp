#include "FrameBuffer.h"
#include <GLFW/glfw3.h>
#include "Utils/UtilCollection.h"


FrameBuffer::FrameBuffer(std::vector<Texture> rendertargets, const bool useDepthStencil, const GLenum renderbufferFormat)
{
    glCreateFramebuffers(1, &m_name);
    bind();
    int attachmentNumber = 0;
    for(const auto& texture : rendertargets)
    {
        glNamedFramebufferTexture(m_name, GL_COLOR_ATTACHMENT0 + attachmentNumber, texture.getName(), 0);
        attachmentNumber++;
    }
    auto fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Framebuffer is not complete!");
    if(useDepthStencil)
    {
        attachDepthStencil(rendertargets.at(0).getWidth(), rendertargets.at(0).getHeight(), renderbufferFormat);
    }
    fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Framebuffer is not complete!");
    unbind();
}

FrameBuffer::FrameBuffer(GLenum attachmentType, Texture depthAttachment)
{
    if (attachmentType != GL_DEPTH_ATTACHMENT)
        throw std::runtime_error("This constructor is for using depth textures only");

    glCreateFramebuffers(1, &m_name);
    bind();
    glNamedFramebufferTexture(m_name, GL_DEPTH_ATTACHMENT, depthAttachment.getName(), 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    unbind();

}

FrameBuffer::FrameBuffer(const int width, const int height, const bool useDepthStencil, const GLenum renderbufferFormat)
{
    glCreateFramebuffers(1, &m_name);
    if (useDepthStencil)
    {
        attachDepthStencil(width, height, renderbufferFormat);
    }
}

FrameBuffer::~FrameBuffer()
{
    if (glfwGetCurrentContext() != nullptr) 
	{
        glDeleteFramebuffers(1, &m_name);
    }
    util::getGLerror(__LINE__, __FUNCTION__);
}

void FrameBuffer::bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_name);
}

void FrameBuffer::unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint FrameBuffer::getName() const
{
    return m_name;
}

void FrameBuffer::attachDepthStencil(const int width, const int height, const GLenum renderbufferFormat)
{
    glCreateRenderbuffers(1, &m_rbo);
    glNamedRenderbufferStorage(m_rbo, renderbufferFormat, width, height);
    if(renderbufferFormat == GL_DEPTH_COMPONENT16 || renderbufferFormat == GL_DEPTH_COMPONENT24 || renderbufferFormat == GL_DEPTH_COMPONENT32)
        glNamedFramebufferRenderbuffer(m_name, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
    else
        glNamedFramebufferRenderbuffer(m_name, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
}
