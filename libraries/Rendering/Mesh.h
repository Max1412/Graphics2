#pragma once
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

class Mesh {
public:
    Mesh(aiMesh* assimpMesh);
    
    const std::vector<glm::vec3>& getVertices() const;
    const std::vector<glm::vec3>& getNormals() const;
    const std::vector<glm::vec3>& getTexCoords() const;
    const std::vector<unsigned int>& getIndices() const;

private:
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec3> m_texCoords;
    std::vector<unsigned int> m_indices;


};