#include "Mesh.h"

Mesh::Mesh(aiMesh* assimpMesh) {

    if (!assimpMesh->HasNormals() ||/* !assimpMesh->HasTextureCoords(0) || */!assimpMesh->HasFaces()) {
        throw std::runtime_error("Mesh must have normals, tex coords, faces");
    }

    m_vertices.resize(assimpMesh->mNumVertices);
    m_normals.resize(assimpMesh->mNumVertices);
    //m_texCoords.resize(assimpMesh->mNumVertices);

    #pragma omp parallel for
    for (int i = 0; i < assimpMesh->mNumVertices; i++) {
        aiVector3D aivec = assimpMesh->mVertices[i];
        glm::vec3 vertex(aivec.x, aivec.y, aivec.z);
        m_vertices.at(i) = vertex;

        aiVector3D ainorm = assimpMesh->mNormals[i];
        glm::vec3 normal(ainorm.x, ainorm.y, ainorm.z);
        m_normals.at(i) = normal;

        /*
        aiVector3D* aitex = assimpMesh->mTextureCoords[i];
        glm::vec3 tex(aitex->x, aitex->y, aitex->z);
        m_texCoords.at(i) = tex;
        */
    }
    for (unsigned int i = 0; i < assimpMesh->mNumFaces; i++) {
        aiFace face = assimpMesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            m_indices.push_back(face.mIndices[j]);
        }
    }
}

const std::vector<glm::vec3>& Mesh::getVertices() const {
    return m_vertices;
}

const std::vector<glm::vec3>& Mesh::getNormals() const {
    return m_normals;
}

const std::vector<glm::vec3>& Mesh::getTexCoords() const {
    return m_texCoords;
}

const std::vector<unsigned int>& Mesh::getIndices() const {
    return m_indices;
}