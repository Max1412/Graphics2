#pragma once

#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <vector>

#include "Buffer.h"
#include "VertexArray.h"

class Mesh
{
public:
    Mesh(aiMesh* assimpMesh, bool useOwnBuffers = true);
    Mesh(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned>& indices);

    /**
     * \brief returns vertices as vector of vec3
     * \return vertices
     */
    const std::vector<glm::vec3>& getVertices() const;

    /**
     * \brief returns normals as vector of vec3
     * \return normals
     */
    const std::vector<glm::vec3>& getNormals() const;

    /**
     * \brief returns UV coords as vector of vec3
     * \return UV/texture coordinates
     */
    const std::vector<glm::vec3>& getTexCoords() const;

    /**
     * \brief returns indices as vectors of uint
     * \return indices
     */
    const std::vector<unsigned>& getIndices() const;

    /**
     * \brief returns the model matrix
     * \return 4x4 model matrix, initially identity matrix
     */
    const glm::mat4& getModelMatrix() const;

    /**
     * \brief returns the material ID
     * \return material ID
     */
    unsigned getMaterialID() const;
    unsigned getMaterialIndex() const;

    /**
    * \brief returns the previously calculated bounding box
    * \return bmin: getBoundingBox()[0], bmax: getBoundingBox()[1]
    */
    const glm::mat2x3& getBoundingBox() const;

    /**
    * \brief re-calculates the bounding box and returns it
    * \return bmin: getBoundingBox()[0], bmax: getBoundingBox()[1]
    */
    const glm::mat2x3& calculateBoundingBox();

    /**
     * \brief calls forceDraw() if the mesh is enabled for rendering
     */
    void draw() const;

    /**
    * \brief binds the vao & index buffer and uses glDrawArrays
    */
    void forceDraw() const;

    /**
     * \brief sets the model matrix
     * \param modelMatrix model matrix for this mesh
     */
    void setModelMatrix(const glm::mat4& modelMatrix);

    /**
     * \brief sets the material ID
     * \param materialID material ID for this mesh
     */
    void setMaterialID(const unsigned materialID);

    void setEnabledForRendering(bool enable);
    bool isEnabledForRendering() const;

private:
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec3> m_texCoords;
    std::vector<unsigned> m_indices;

    bool m_enabledForRendering = true;

    glm::mat2x3 m_boundingBox;

    glm::mat4 m_modelMatrix = glm::mat4(1.0f);

    unsigned m_materialID = 1U;

    Buffer m_vertexBuffer;
    Buffer m_normalBuffer;
    Buffer m_texCoordBuffer;
    Buffer m_indexBuffer;
    VertexArray m_vao;

    unsigned int m_materialIndex;
};
