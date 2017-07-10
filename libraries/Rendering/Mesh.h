#pragma once
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

class Mesh {
public:
    Mesh(aiMesh* assimpMesh);
    
    std::vector<glm::vec3> getVertices();
    std::vector<glm::vec3> getNormals();
    std::vector<glm::vec3> getTexCoords();
    std::vector<unsigned int> getIndices();

private:
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec3> m_texCoords;
    std::vector<unsigned int> m_indices;


};