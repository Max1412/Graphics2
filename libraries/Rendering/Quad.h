#pragma once

#include "Mesh.h"

class Quad
{
public:
    Quad();
    void draw() const;
private:
    VertexArray m_quadVAO;
    Buffer m_quadBuffer;
    Buffer m_texCoordBuffer;

    static std::vector<glm::vec2> quadVertices;
    static std::vector<glm::vec2> quadTexCoords;
};