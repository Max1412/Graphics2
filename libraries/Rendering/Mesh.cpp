#include "Mesh.h"

Mesh::Mesh(aiMesh* assimpMesh) {

    if (!assimpMesh->HasNormals() ||/* !assimpMesh->HasTextureCoords(0) || */!assimpMesh->HasFaces()) {
        throw std::runtime_error("Mesh must have normals, tex coords, faces");
    }

    for (unsigned int i = 0; i < assimpMesh->mNumVertices; i++) {
        aiVector3D aivec = assimpMesh->mVertices[i];
        glm::vec3 vertex(aivec.x, aivec.y, aivec.z);
        m_vertices.push_back(vertex);

        aiVector3D ainorm = assimpMesh->mNormals[i];
        glm::vec3 normal(ainorm.x, ainorm.y, ainorm.z);
        m_normals.push_back(normal);
        /*
        aiVector3D* aitex = assimpMesh->mTextureCoords[i];
        glm::vec3 tex(aitex->x, aitex->y, aitex->z);
        m_texCoords.push_back(tex);
        */
    }
    for (unsigned int i = 0; i < assimpMesh->mNumFaces; i++) {
        aiFace face = assimpMesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            m_indices.push_back(face.mIndices[j]);
        }
    }
}

std::vector<glm::vec3> Mesh::getVertices() {
    return m_vertices;
}

std::vector<glm::vec3> Mesh::getNormals() {
    return m_normals;
}

std::vector<glm::vec3> Mesh::getTexCoords() {
    return m_texCoords;
}

std::vector<unsigned int> Mesh::getIndices() {
    return m_indices;
}