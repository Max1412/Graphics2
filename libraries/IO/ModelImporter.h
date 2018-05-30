#pragma once

#include <vector>
#include <memory>
#include <experimental/filesystem>
#include <unordered_map>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include "Rendering/Mesh.h"
#include "Rendering/Texture.h"
#include "Rendering/Uniform.h"
#include "Rendering/Camera.h"
#include "Rendering/ShaderProgram.h"

class ShaderProgram;

struct PhongGPUMaterial
{
    uint64_t diffTexture = -1;
    uint64_t specTexture = -1;
    uint64_t opacityTexture = -1;
    float opacity = 1.0f;
    float Ns = -1.0f;
    glm::vec4 diffColor;
    glm::vec4 specColor;
    glm::vec4 emissiveColor;
};

struct Indirect
{
    unsigned count;
    unsigned instanceCount;
    unsigned firstIndex;
    unsigned baseVertex;
    unsigned baseInstance;
};

class ModelImporter
{
public:
    explicit ModelImporter(const std::experimental::filesystem::path& filename);
    ModelImporter(const std::experimental::filesystem::path& filename, int test);

    std::vector<std::shared_ptr<Mesh>> getMeshes() const;

    void draw(const ShaderProgram& sp) const;
    void multiDraw(const ShaderProgram& sp) const;
    void multiDrawCulled(const ShaderProgram & sp, const glm::mat4 & viewProjection) const;
    void drawCulled(const ShaderProgram& sp, const glm::mat4& view, float angle, float ratio, float near, float far) const;

    void registerUniforms(ShaderProgram& sp) const;
    void resetIndirectDrawParams();

private:
    Assimp::Importer m_importer;
    const aiScene* m_scene;
    std::vector<std::shared_ptr<Mesh>> m_meshes;
    std::unordered_map<std::string, std::shared_ptr<Texture>> m_texturemap;
    std::vector<PhongGPUMaterial> m_gpuMaterials;
    Buffer m_gpuMaterialBuffer;

    std::vector<unsigned> m_gpuMaterialIndices;
    Buffer m_gpuMaterialIndicesBuffer;

    std::vector<glm::mat4> m_modelMatrices;
    Buffer m_modelMatrixBuffer;

    std::shared_ptr<Uniform<int>> m_meshIndexUniform;
    std::shared_ptr<Uniform<int>> m_materialIndexUniform;

    // multi-draw buffers
    std::vector<unsigned> m_allTheIndices;
    std::vector<glm::vec3> m_allTheVertices;
    std::vector<glm::vec3> m_allTheNormals;
    std::vector<glm::vec3> m_allTheTexCoords;

    std::vector<Indirect> m_indirectDrawParams;
    Buffer m_indirectDrawBuffer;

    Buffer m_multiDrawIndexBuffer;
    Buffer m_multiDrawVertexBuffer;
    Buffer m_multiDrawNormalBuffer;
    Buffer m_multiDrawTexCoordBuffer;
    VertexArray m_multiDrawVao;

    // culling stuff
    std::vector<glm::mat2x4> m_boundingBoxes;
    Buffer m_boundingBoxBuffer;
    std::shared_ptr<Uniform<glm::mat4>> m_viewProjUniform;
    ShaderProgram m_cullingProgram;
};
