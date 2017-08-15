#pragma once

#include <vector>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include "Rendering/Mesh.h"

class ModelImporter {
public:
	ModelImporter(const std::string &filename);

    std::vector<std::shared_ptr<Mesh>> getMeshes();

private:
	Assimp::Importer m_importer;
	const aiScene* m_scene;
	std::vector<std::shared_ptr<Mesh>> m_meshes;

};
