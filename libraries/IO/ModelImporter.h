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

class ShaderProgram;

struct PhongGPUMaterial
{
    uint64_t diffTexture = -1;
    uint64_t specTexture = -1;
    glm::vec4 diffColor;
    glm::vec4 specColor;
    glm::vec3 emissiveColor;
    float Ns = -1.0f;
};


class ModelImporter
{
public:
    explicit ModelImporter(const std::experimental::filesystem::path& filename);
    ModelImporter(const std::experimental::filesystem::path& filename, int test);

    std::vector<std::shared_ptr<Mesh>> getMeshes() const;

    void draw(const ShaderProgram& sp) const;

    void registerUniforms(ShaderProgram& sp) const;

private:
    Assimp::Importer m_importer;
    const aiScene* m_scene;
    std::vector<std::shared_ptr<Mesh>> m_meshes;
    std::unordered_map<std::string, std::shared_ptr<Texture>> m_texturemap;
    std::vector<PhongGPUMaterial> m_gpuMaterials;
    Buffer m_gpuMaterialBuffer;

    std::vector<glm::mat4> m_modelMatrices;
    Buffer m_modelMatrixBuffer;

    std::shared_ptr<Uniform<int>> m_meshIndexUniform;
    std::shared_ptr<Uniform<int>> m_materialIndexUniform;

};
