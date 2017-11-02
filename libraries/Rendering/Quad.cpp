#include "Quad.h"

Quad::Quad() : m_quadBuffer(GL_ARRAY_BUFFER), m_texCoordBuffer(GL_ARRAY_BUFFER)
{
    m_quadBuffer.setData(quadVertices, GL_STATIC_DRAW);

    m_texCoordBuffer.setData(quadTexCoords, GL_STATIC_DRAW);

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