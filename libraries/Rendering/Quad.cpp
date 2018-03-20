#include "Quad.h"

Quad::Quad() : m_quadBuffer(GL_ARRAY_BUFFER), m_texCoordBuffer(GL_ARRAY_BUFFER)
{
    m_quadBuffer.setStorage(quadVertices, GL_DYNAMIC_STORAGE_BIT);

    m_texCoordBuffer.setStorage(quadTexCoords, GL_DYNAMIC_STORAGE_BIT);

    m_quadVAO.connectBuffer(m_quadBuffer, 0, 2, GL_FLOAT, GL_FALSE);
    m_quadVAO.connectBuffer(m_texCoordBuffer, 1, 2, GL_FLOAT, GL_FALSE);
}

void Quad::draw() const
{
    m_quadVAO.bind();
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(quadVertices.size()));
}

std::vector<glm::vec2> Quad::quadVertices = {
    { -1.0, -1.0 },
    { 1.0, -1.0 },
    { -1.0, 1.0 },
    { -1.0, 1.0 },
    { 1.0, -1.0 },
    { 1.0, 1.0 }
};

std::vector<glm::vec2> Quad::quadTexCoords = {
    { 0.0, 0.0 },
    { 1.0, 0.0 },
    { 0.0, 1.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 }
};