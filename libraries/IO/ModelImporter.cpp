#include "ModelImporter.h"

#include <iostream>

#include <assimp/postprocess.h>

ModelImporter::ModelImporter(const std::experimental::filesystem::path& filename)
{
    const auto path = util::gs_resourcesPath / filename;
    const auto pathString = path.string();
    m_scene = m_importer.ReadFile(pathString.c_str(), aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_JoinIdenticalVertices);

    if (!m_scene || m_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
    {
        const std::string err = m_importer.GetErrorString();
        throw std::runtime_error("Assimp import failed: " + err);
    }
    std::cout << "Model succesfully loaded from " << filename.string() << std::endl;

    if (m_scene->HasMeshes())
    {
        const auto numMeshes = m_scene->mNumMeshes;
        m_meshes.reserve(numMeshes);
        for (unsigned i = 0; i < numMeshes; i++)
        {
            m_meshes.emplace_back(std::make_shared<Mesh>(m_scene->mMeshes[i]));
        }
    }
}

std::vector<std::shared_ptr<Mesh>> ModelImporter::getMeshes() const
{
    return m_meshes;
}
