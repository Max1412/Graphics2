#include "ModelImporter.h"

ModelImporter::ModelImporter(const std::string &filename) {
	m_importer;
	m_scene = m_importer.ReadFile(RESOURCES_PATH + std::string("/") + filename,
		aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_FlipWindingOrder | aiProcess_GenUVCoords);

	if (!m_scene || m_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
		std::string err = m_importer.GetErrorString();
		throw std::runtime_error("Assimp import failed: " + err);
	}
	else {
		std::cout << "Model succesfully loaded from " << filename << std::endl;
	}

	if (m_scene->HasMeshes()) {
		const unsigned int numMeshes = m_scene->mNumMeshes;
		for (int i = 0; i < numMeshes; i++) {
			m_meshes.push_back(Mesh(m_scene->mMeshes[i]));
		}
	}
}

ModelImporter::~ModelImporter() {
}

std::vector<Mesh> ModelImporter::getMeshes() {
	return m_meshes;
}