#pragma once

#include <vector>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include "Rendering/Mesh.h"
#include <filesystem>

class ModelImporter {
public:
	explicit ModelImporter(const std::experimental::filesystem::path& filename);

    std::vector<std::shared_ptr<Mesh>> getMeshes() const;

private:
	Assimp::Importer m_importer;
	const aiScene* m_scene;
	std::vector<std::shared_ptr<Mesh>> m_meshes;

};
