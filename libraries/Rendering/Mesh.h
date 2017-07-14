#pragma once
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

#include "Buffer.h"
#include "VertexArray.h"

class Mesh {
public:
    Mesh(aiMesh* assimpMesh);
    Mesh(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned>& indices);

    void del();
    
    const std::vector<glm::vec3>& getVertices() const;
    const std::vector<glm::vec3>& getNormals() const;
    const std::vector<glm::vec3>& getTexCoords() const;
    const std::vector<unsigned int>& getIndices() const;

    const glm::mat4& getModelMatrix() const;
    const unsigned getMaterialID() const;


    void draw() const;

    void setModelMatrix(const glm::mat4& modelMatrix);
    void setMaterialID(const unsigned materialID);


private:
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec3> m_texCoords;
    std::vector<unsigned int> m_indices;

    glm::mat4 m_modelMatrix = glm::mat4(1.0f);

    unsigned m_materialID = 1U;

    Buffer m_vertexBuffer;
    Buffer m_normalBuffer;
    Buffer m_indexBuffer;
    VertexArray m_vao;
};