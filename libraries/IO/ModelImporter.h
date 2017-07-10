#pragma once

#include <string>
#include <iostream>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Rendering/Mesh.h"

class ModelImporter {
public:
	ModelImporter(const std::string &filename);
	~ModelImporter();

	std::vector<Mesh> getMeshes();

private:
	Assimp::Importer m_importer;
	const aiScene* m_scene;
	std::vector<Mesh> m_meshes;

};