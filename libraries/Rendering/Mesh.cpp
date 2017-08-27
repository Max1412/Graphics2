#include "Mesh.h"
#include <GLFW/glfw3.h>

Mesh::Mesh(aiMesh* assimpMesh) : m_vertexBuffer(GL_ARRAY_BUFFER), m_normalBuffer(GL_ARRAY_BUFFER), m_indexBuffer(GL_ELEMENT_ARRAY_BUFFER) {

    if (!assimpMesh->HasNormals() ||/* !assimpMesh->HasTextureCoords(0) || */!assimpMesh->HasFaces()) {
        throw std::runtime_error("Mesh must have normals, tex coords, faces");
    }

    m_vertices.resize(assimpMesh->mNumVertices);
    m_normals.resize(assimpMesh->mNumVertices);
    //m_texCoords.resize(assimpMesh->mNumVertices);

    #pragma omp parallel for
    for (auto i = 0; i < assimpMesh->mNumVertices; i++) {
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

    m_vertexBuffer.setData(m_vertices, GL_STATIC_DRAW);
    m_normalBuffer.setData(m_normals, GL_STATIC_DRAW);
    m_indexBuffer.setData(m_indices, GL_STATIC_DRAW);
    m_vao.connectBuffer(m_vertexBuffer, 0, 3, GL_FLOAT, GL_FALSE);
    m_vao.connectBuffer(m_normalBuffer, 1, 3, GL_FLOAT, GL_FALSE);
    m_vao.connectIndexBuffer(m_indexBuffer);
}

Mesh::Mesh(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned>& indices)
        : m_vertices(vertices), m_normals(normals), m_indices(indices),
        m_vertexBuffer(GL_ARRAY_BUFFER), m_normalBuffer(GL_ARRAY_BUFFER), m_indexBuffer(GL_ELEMENT_ARRAY_BUFFER)
{
    m_vertexBuffer.setData(m_vertices, GL_STATIC_DRAW);
    m_normalBuffer.setData(m_normals, GL_STATIC_DRAW);
    m_indexBuffer.setData(m_indices, GL_STATIC_DRAW);
    m_vao.connectBuffer(m_vertexBuffer, 0, 3, GL_FLOAT, GL_FALSE);
    m_vao.connectBuffer(m_normalBuffer, 1, 3, GL_FLOAT, GL_FALSE);
    m_vao.connectIndexBuffer(m_indexBuffer);
}

void Mesh::draw() const {
    m_vao.bind();
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_indices.size()), GL_UNSIGNED_INT, nullptr);
}


void Mesh::setModelMatrix(const glm::mat4& modelMatrix) {
    m_modelMatrix = modelMatrix;
}

const glm::mat4& Mesh::getModelMatrix() const {
    return m_modelMatrix;
}

const unsigned Mesh::getMaterialID() const {
    return m_materialID;
}

void Mesh::setMaterialID(const unsigned materialID) {
    m_materialID = materialID;
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