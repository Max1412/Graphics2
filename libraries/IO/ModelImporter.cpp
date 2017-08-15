#include "ModelImporter.h"

#include <iostream>

#include <assimp/postprocess.h>

ModelImporter::ModelImporter(const std::string &filename) {
	m_scene = m_importer.ReadFile(RESOURCES_PATH + std::string("/") + filename,
		aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_GenUVCoords);

	if (!m_scene || m_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
		std::string err = m_importer.GetErrorString();
		throw std::runtime_error("Assimp import failed: " + err);
	}
	else {
		std::cout << "Model succesfully loaded from " << filename << std::endl;
	}

	if (m_scene->HasMeshes()) {
		const auto numMeshes = m_scene->mNumMeshes;
        m_meshes.reserve(numMeshes);
		for (unsigned i = 0; i < numMeshes; i++) {
			m_meshes.emplace_back(std::make_shared<Mesh>(m_scene->mMeshes[i]));
		}
	}
}

std::vector<std::shared_ptr<Mesh>> ModelImporter::getMeshes() {
	return m_meshes;
}