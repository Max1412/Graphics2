#include "ModelImporter.h"

#include <iostream>

#include <assimp/postprocess.h>

#include "stb/stb_image.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Binding.h"
#include <execution>
#include <algorithm>
#include <unordered_set>

ModelImporter::ModelImporter(const std::experimental::filesystem::path& filename)
    : m_gpuMaterialBuffer(GL_SHADER_STORAGE_BUFFER), m_gpuMaterialIndicesBuffer(GL_SHADER_STORAGE_BUFFER), m_modelMatrixBuffer(GL_SHADER_STORAGE_BUFFER),
    m_indirectDrawBuffer(GL_DRAW_INDIRECT_BUFFER), m_multiDrawIndexBuffer(GL_ELEMENT_ARRAY_BUFFER), m_multiDrawVertexBuffer(GL_ARRAY_BUFFER), 
    m_multiDrawNormalBuffer(GL_ARRAY_BUFFER), m_multiDrawTexCoordBuffer(GL_ARRAY_BUFFER), m_boundingBoxBuffer(GL_SHADER_STORAGE_BUFFER),
    m_cullingProgram({ Shader("frustumCulling.comp", GL_COMPUTE_SHADER, BufferBindings::g_definitions) })
{

    m_meshIndexUniform = std::make_shared<Uniform<int>>("meshIndex", -1);
    m_materialIndexUniform = std::make_shared<Uniform<int>>("materialIndex", -1);

    const auto path = util::gs_resourcesPath / filename;
    const auto pathString = path.string();

    m_scene = m_importer.ReadFile(pathString.c_str(), aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_JoinIdenticalVertices);

    if (!m_scene || m_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
    {
        const std::string err = m_importer.GetErrorString();
        throw std::runtime_error("Assimp import failed: " + err);
    }
    std::cout << "Loading model from " << filename.string() << std::endl;

    if (m_scene->HasMeshes())
    {
        const auto numMeshes = m_scene->mNumMeshes;
        m_meshes.reserve(numMeshes);
        m_modelMatrices = std::vector<glm::mat4>(numMeshes, glm::mat4(1.0f));
        for (unsigned i = 0; i < numMeshes; i++)
        {
            m_meshes.emplace_back(std::make_shared<Mesh>(m_scene->mMeshes[i]));
        }
    }

    const auto root = m_scene->mRootNode;
    const glm::mat4 startTransform(1.0f);

    static_assert(alignof(aiMatrix4x4) == alignof(glm::mat4) && sizeof(aiMatrix4x4) == sizeof(glm::mat4));

    std::function<void(aiNode* node, glm::mat4 trans)> traverseChildren = [this, &traverseChildren](aiNode* node, glm::mat4 trans)
    {
        // check if transformation exists
        if (std::none_of(&node->mTransformation.a1, (&node->mTransformation.d4) + 1,
            [](float f) { return std::isnan(f) || std::isinf(f); }))
        {
            // accumulate transform
            const glm::mat4 transform = reinterpret_cast<glm::mat4&>(node->mTransformation);
            trans *= transform;
        }
        // TODO what to do with missing transformation

        // assign transformation to meshes
        for (unsigned i = 0; i < node->mNumMeshes; i++)
        {
            m_meshes.at(node->mMeshes[i])->setModelMatrix(trans);
            m_modelMatrices.at(node->mMeshes[i]) = trans;
        }

        // recursively work on the child nodes
        for (unsigned i = 0; i < node->mNumChildren; i++)
        {
            traverseChildren(node->mChildren[i], trans);
        }
    };

    traverseChildren(root, startTransform);

    if (m_scene->HasMaterials())
    {
        const auto numMaterials = m_scene->mNumMaterials;
        for (unsigned i = 0; i < numMaterials; i++)
        {
            const auto mat = m_scene->mMaterials[i];

            PhongGPUMaterial gpuMat;

            // use those to encode the presence of a texture in the alpha channel
            float hasDiff = -1.0f;
            float hasSpec = -1.0f;

            aiString reltexPath;
            // TODO emissive, ... (use texture count and switch/case to take all of them)
            for (aiTextureType type : {aiTextureType_DIFFUSE, aiTextureType_SPECULAR, aiTextureType_OPACITY, aiTextureType_HEIGHT, aiTextureType_NORMALS})
            {
                if (mat->GetTextureCount(type) == 0)
                {
                    switch (type)
                    {
                    case aiTextureType_DIFFUSE:
                        hasDiff = 0.0f;
                        break;
                    case aiTextureType_SPECULAR:
                        hasSpec = 0.0f;
                        break;
                    case aiTextureType_OPACITY:
                        gpuMat.opacity = 1.0f;
                        break;
                    default:
                        break;
                    }
                    continue;
                }

                uint64_t texID = std::numeric_limits<uint64_t>::max();
                mat->GetTexture(type, 0, &reltexPath);
                auto absTexPath = path.parent_path() / std::experimental::filesystem::path(reltexPath.C_Str());

                // texture not loaded yet
                if (m_texturemap.count(reltexPath.C_Str()) == 0)
                {
                    auto tex = std::make_shared<Texture>(GL_TEXTURE_2D, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
                    // TODO textures with less than 4 channels... detect automatically?
                    if (stbi_is_hdr(absTexPath.string().c_str()))
                    {
                        tex->loadFromFile(absTexPath, GL_RGBA32F, GL_RGBA, GL_FLOAT, STBI_rgb_alpha);
                    }
                    else if(type == aiTextureType_OPACITY || type == aiTextureType_HEIGHT)
                    {
                        tex->loadFromFile(absTexPath, GL_R8, GL_RED, GL_UNSIGNED_BYTE, STBI_grey);
                    }
                    else if (type == aiTextureType_NORMALS)
                    {
                        tex->loadFromFile(absTexPath, GL_RGB16F, GL_RGB, GL_UNSIGNED_BYTE, STBI_rgb);
                    }
                    else
                    {
                        tex->loadFromFile(absTexPath);
                    }
                    texID = tex->generateHandle();
                    tex->generateMipmap();
                    m_texturemap.emplace(reltexPath.C_Str(), tex);
                }
                else // texture already loaded, store handle
                {
                    texID = m_texturemap.at(reltexPath.C_Str())->getHandle();
                }

                switch (type)
                {
                    case aiTextureType_DIFFUSE:
                        gpuMat.diffTexture = texID;
                        if(texID == std::numeric_limits<uint64_t>::max())
                        {
                            hasDiff = 0.0f;
                        }
                        break;
                    case aiTextureType_SPECULAR:
                        gpuMat.specTexture = texID;
                        if (texID == std::numeric_limits<uint64_t>::max())
                        {
                            hasSpec = 0.0f;
                        }
                        break;
                    case aiTextureType_HEIGHT:
                        gpuMat.bumpTexture = texID;
                        gpuMat.bumpType = 2;
                        break;
                    case aiTextureType_NORMALS:
                        gpuMat.bumpTexture = texID;
                        gpuMat.bumpType = 1;
                        break;
                    case aiTextureType_OPACITY:
                        gpuMat.opacityTexture = texID;
                        if (texID == std::numeric_limits<uint64_t>::max())
                        {
                            gpuMat.opacity = 1.0f;
                        }
                        else
                        {
                            // encode "having an opacity texture" like this
                            gpuMat.opacity = -1.0f;
                        }
                        break;
                    default:
                        break;
                }
            }

            aiColor3D diffcolor(0.0f, 1.0f, 0.0f);
            mat->Get(AI_MATKEY_COLOR_DIFFUSE, diffcolor);
            aiColor3D speccolor(0.0f, 1.0f, 0.0f);
            mat->Get(AI_MATKEY_COLOR_SPECULAR, speccolor);
            aiColor3D emissivecolor(0.0f, 1.0f, 0.0f);
            mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissivecolor);
            float Ns = -1.0f;
            mat->Get(AI_MATKEY_SHININESS, Ns);

            if(gpuMat.opacity != -1.0f) // only load opacity if no opacity texture exists
            {
                mat->Get(AI_MATKEY_OPACITY, gpuMat.opacity);   
            }

            gpuMat.diffColor = glm::vec4(diffcolor.r, diffcolor.g, diffcolor.b, hasDiff);
            gpuMat.specColor = glm::vec4(speccolor.r, speccolor.g, speccolor.b, hasSpec);
            gpuMat.emissiveColor = glm::vec4(emissivecolor.r, emissivecolor.g, emissivecolor.b, 1.0f);
            gpuMat.Ns = Ns;
            //if (Ns != 0)
            //    gpuMat.Ns = Ns;
            //else
            //    gpuMat.Ns = 32.0f;

            m_gpuMaterials.push_back(gpuMat);
        }
    }

    m_gpuMaterialBuffer.setStorage(m_gpuMaterials, GL_DYNAMIC_STORAGE_BIT);
    m_gpuMaterialBuffer.bindBase(BufferBindings::Binding::materials);

    m_modelMatrixBuffer.setStorage(m_modelMatrices, GL_DYNAMIC_STORAGE_BIT);
    m_modelMatrixBuffer.bindBase(BufferBindings::Binding::modelMatrices);

	std::vector<std::shared_ptr<Mesh>> transparentMeshes;
	for (int i = 0; i < m_meshes.size(); i++)
	{
		if (PhongGPUMaterial mat = m_gpuMaterials.at(m_meshes.at(i)->getMaterialIndex()); mat.opacityTexture != -1 && mat.opacity != 1)
		{
			transparentMeshes.push_back(m_meshes.at(i));
			m_meshes.erase(m_meshes.begin() + i--);
		}
	}
	m_meshes.insert(m_meshes.end(), transparentMeshes.begin(), transparentMeshes.end());

    std::cout << "Loading complete: " << filename.string() << std::endl;

	unsigned start = 0;
    unsigned baseVertexOffset = 0;
    for (const auto& mesh : m_meshes)
    {
        m_gpuMaterialIndices.push_back(mesh->getMaterialIndex());
        m_boundingBoxes.emplace_back(mesh->getBoundingBox());

        m_allTheIndices.insert(m_allTheIndices.end(), mesh->getIndices().begin(), mesh->getIndices().end());
        m_allTheVertices.insert(m_allTheVertices.end(), mesh->getVertices().begin(), mesh->getVertices().end());
        m_allTheNormals.insert(m_allTheNormals.end(), mesh->getNormals().begin(), mesh->getNormals().end());
        m_allTheTexCoords.insert(m_allTheTexCoords.end(), mesh->getTexCoords().begin(), mesh->getTexCoords().end());

        const auto count = static_cast<unsigned>(mesh->getIndices().size());

        m_indirectDrawParams.push_back({ count, 1U, start, baseVertexOffset, 0U });

        start += static_cast<unsigned>(mesh->getIndices().size());
        baseVertexOffset += static_cast<unsigned>(mesh->getVertices().size());
    }

    m_gpuMaterialIndices.shrink_to_fit();
    m_boundingBoxes.shrink_to_fit();
    m_allTheIndices.shrink_to_fit();
    m_allTheVertices.shrink_to_fit();
    m_allTheNormals.shrink_to_fit();
    m_allTheTexCoords.shrink_to_fit();
    m_indirectDrawParams.shrink_to_fit();

    m_gpuMaterialIndicesBuffer.setStorage(m_gpuMaterialIndices, GL_DYNAMIC_STORAGE_BIT);
    m_gpuMaterialIndicesBuffer.bindBase(BufferBindings::Binding::materialIndices);

    m_boundingBoxBuffer.setStorage(m_boundingBoxes, GL_DYNAMIC_STORAGE_BIT); //TODO: padding correct?
    m_boundingBoxBuffer.bindBase(static_cast<BufferBindings::Binding>(6));

    m_viewProjUniform = std::make_shared<Uniform<glm::mat4>>("viewProjMatrix", glm::mat4(1.0f));
    m_cullingProgram.addUniform(m_viewProjUniform);

    m_indirectDrawBuffer.setStorage(m_indirectDrawParams, GL_DYNAMIC_STORAGE_BIT);

    m_multiDrawIndexBuffer.setStorage(m_allTheIndices, GL_DYNAMIC_STORAGE_BIT);
    m_multiDrawVertexBuffer.setStorage(m_allTheVertices, GL_DYNAMIC_STORAGE_BIT);
    m_multiDrawNormalBuffer.setStorage(m_allTheNormals, GL_DYNAMIC_STORAGE_BIT);
    m_multiDrawTexCoordBuffer.setStorage(m_allTheTexCoords, GL_DYNAMIC_STORAGE_BIT);

    m_multiDrawVao.connectBuffer(m_multiDrawVertexBuffer, BufferBindings::VertexAttributeLocation::vertices, 3, GL_FLOAT, GL_FALSE);
    m_multiDrawVao.connectBuffer(m_multiDrawNormalBuffer, BufferBindings::VertexAttributeLocation::normals, 3, GL_FLOAT, GL_FALSE);
    m_multiDrawVao.connectBuffer(m_multiDrawTexCoordBuffer, BufferBindings::VertexAttributeLocation::texCoords, 3, GL_FLOAT, GL_FALSE);

    m_multiDrawVao.connectIndexBuffer(m_multiDrawIndexBuffer);
}

std::vector<std::shared_ptr<Mesh>> ModelImporter::loadAllMeshesFromFile(const std::experimental::filesystem::path& filename)
{
    const auto path = util::gs_resourcesPath / filename;
    const auto pathString = path.string();
    
    Assimp::Importer importer;
    std::vector<std::shared_ptr<Mesh>> meshes;

    const aiScene * scene = importer.ReadFile(pathString.c_str(), aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_JoinIdenticalVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
    {
        const std::string err = importer.GetErrorString();
        throw std::runtime_error("Assimp import failed: " + err);
    }
    std::cout << "Model succesfully loaded from " << filename.string() << std::endl;

    if (scene->HasMeshes())
    {
        const auto numMeshes = scene->mNumMeshes;
        meshes.reserve(numMeshes);
        for (unsigned i = 0; i < numMeshes; i++)
        {
            meshes.emplace_back(std::make_shared<Mesh>(scene->mMeshes[i]));
        }
    }

    return meshes;
}

void ModelImporter::registerUniforms(ShaderProgram& sp) const
{
    try
    {
        sp.addUniform(m_meshIndexUniform);
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what() << '\n';
        std::cout << "WARNING: No Mesh Index Uniform avaiable. TODO: make this selectable for multidraw\n";
    }
    try
    {
        sp.addUniform(m_materialIndexUniform);
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what() << '\n';
        std::cout << "WARNING: No Material Index Uniform avaiable. TODO: make this selectable for multidraw\n";
    }
}

void ModelImporter::resetIndirectDrawParams()
{
    m_indirectDrawBuffer.setContentToContainerSubData(m_indirectDrawParams, 0);
}

void ModelImporter::draw(const ShaderProgram& sp) const
{
    sp.use();
    int i = 0;
    for(const auto& mesh : m_meshes)
    {
        m_meshIndexUniform->setContent(i);
        m_materialIndexUniform->setContent(mesh->getMaterialIndex());
        sp.updateUniforms();
        mesh->forceDraw();
        i++;
    }
}

void ModelImporter::multiDraw(const ShaderProgram& sp) const
{
    //m_materialIndexUniform->setContent(m_meshes.at(0)->getMaterialID());
    sp.use();
    m_multiDrawVao.bind();
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_indirectDrawBuffer.getHandle());
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(m_indirectDrawParams.size()), 0);
    //glMultiDrawElementsBaseVertex(GL_TRIANGLES, m_counts.data(), GL_UNSIGNED_INT, reinterpret_cast<const GLvoid* const*>(m_starts.data()), static_cast<GLsizei>(m_counts.size()), m_baseVertexOffsets.data());
}

void ModelImporter::multiDrawCulled(const ShaderProgram& sp, const glm::mat4& viewProjection) const
{
    // C U L L I N G
    m_viewProjUniform->setContent(viewProjection);
    m_cullingProgram.use();

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_indirectDrawBuffer.getHandle());
    m_boundingBoxBuffer.bindBase(static_cast<BufferBindings::Binding>(6));
    m_modelMatrixBuffer.bindBase(BufferBindings::Binding::modelMatrices);

    glDispatchCompute(static_cast<GLuint>(glm::ceil(m_indirectDrawParams.size() / 64.0f)), 1, 1);
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    // D R A W
    sp.use();
    m_multiDrawVao.bind();
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_indirectDrawBuffer.getHandle());
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(m_indirectDrawParams.size()), 0);
}

void ModelImporter::drawCulled(const ShaderProgram& sp, const glm::mat4& view, float angle, float ratio, float near, float far) const
{
    const glm::vec3 p = glm::inverse(view)[3];
    const glm::vec3 z = glm::normalize(glm::inverse(view)[2]);

    const float tang = glm::tan(angle * 0.5f);
    const float nh = near * tang;
    const float nw = nh * ratio;
    const float fh = far * tang;
    const float fw = fh * ratio;

    // X axis of camera with given "up" vector and Z axis
    const glm::vec3 x = glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), z));

    // the real "up" vector is the cross product of Z and X
    const glm::vec3 y = glm::cross(z, x);

    // compute the centers of the near and far planes
    const glm::vec3 nc = p - z * near;
    const glm::vec3 fc = p - z * far;

    // compute the 4 corners of the frustum on the near plane
    const glm::vec3 ntl = nc + y * nh - x * nw;
    const glm::vec3 ntr = nc + y * nh + x * nw;
    const glm::vec3 nbl = nc - y * nh - x * nw;
    const glm::vec3 nbr = nc - y * nh + x * nw;

    // compute the 4 corners of the frustum on the far plane
    const glm::vec3 ftl = fc + y * fh - x * fw;
    const glm::vec3 ftr = fc + y * fh + x * fw;
    const glm::vec3 fbl = fc - y * fh - x * fw;
    const glm::vec3 fbr = fc - y * fh + x * fw;

    // compute the six planes
    // the function setNormalFromPoints assumes that the points
    // are given in counter clockwise order
    FrustumGeo f;
    f.setNormalFromPoints(FrustumGeo::TOP, ntr, ntl, ftl);
    f.setNormalFromPoints(FrustumGeo::BOTTOM, nbl, nbr, fbr);
    f.setNormalFromPoints(FrustumGeo::LEFT, ntl, nbl, fbl);
    f.setNormalFromPoints(FrustumGeo::RIGHT, nbr, ntr, fbr);
    f.setNormalFromPoints(FrustumGeo::NEAR, ntl, ntr, nbr);
    f.setNormalFromPoints(FrustumGeo::FAR, ftr, ftl, fbl);

    // set frustum vertices
    // caution! points have to lay on their respective planes (by index) 
    f.points = { ntr, nbl, ntl, fbr, nbr, ftr, ftl, fbl };

    const auto cullFunc = [&f](const auto& mesh)
    {
        mesh->setEnabledForRendering(true);

        const glm::vec3 bmin = mesh->getBoundingBox()[0];
        const glm::vec3 bmax = mesh->getBoundingBox()[1];

        bool out;

        // clip bounding box vertices on frustum planes
        for (int i = 0; i < 6; ++i)
        {
            out = true;
            out &= (f.distance(i, glm::vec3(bmin.x, bmin.y, bmin.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmax.x, bmin.y, bmin.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmin.x, bmax.y, bmin.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmax.x, bmax.y, bmin.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmin.x, bmin.y, bmax.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmax.x, bmin.y, bmax.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmin.x, bmax.y, bmax.z)) < 0.0f);
            out &= (f.distance(i, glm::vec3(bmax.x, bmax.y, bmax.z)) < 0.0f);
            if (out)
            {
                mesh->setEnabledForRendering(false);
                return;
            }
        }

        // clip frustum vertices on bounding boxes (needed for some edge-cases)
        for (int axis = 0; axis < 3; ++axis)
        {
            out = true; for (int i = 0; i < 8; i++) out &= (f.points[i][axis] > bmax[axis]); if (out) { mesh->setEnabledForRendering(false); return; }
            out = true; for (int i = 0; i < 8; i++) out &= (f.points[i][axis] < bmin[axis]); if (out) { mesh->setEnabledForRendering(false); return; }
        }
        
    };

    std::for_each(std::execution::par, m_meshes.begin(), m_meshes.end(), cullFunc);

    sp.use();
    int i = 0;
    for (const auto& mesh : m_meshes)
    {
        if (mesh->isEnabledForRendering())
        {
            m_meshIndexUniform->setContent(i);
            m_materialIndexUniform->setContent(mesh->getMaterialIndex());
            sp.updateUniforms();
            mesh->forceDraw();
        }
        i++;
    }
}

std::vector<std::shared_ptr<Mesh>> ModelImporter::getMeshes() const
{
    return m_meshes;
}
