#include "ModelImporter.h"

#include <iostream>

#include <assimp/postprocess.h>

#include "stb/stb_image.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Binding.h"
#include <execution>
#include <algorithm>
#include <unordered_set>

ModelImporter::ModelImporter(const std::experimental::filesystem::path& filename, int test)
: m_gpuMaterialBuffer(GL_SHADER_STORAGE_BUFFER), m_modelMatrixBuffer(GL_SHADER_STORAGE_BUFFER)
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
            // TODO height, opacity, normal, emissive, ... (use texture count and switch/case to take all of them)
            for (aiTextureType type : {aiTextureType_DIFFUSE, aiTextureType_SPECULAR, aiTextureType_OPACITY})
            {
                if (mat->GetTextureCount(type) == 0)
                    continue;

                uint64_t texID = -1;
                mat->GetTexture(type, 0, &reltexPath);
                auto absTexPath = path.parent_path() / std::experimental::filesystem::path(reltexPath.C_Str());

                // texture not loaded yet
                if (m_texturemap.count(reltexPath.C_Str()) == 0)
                {
                    auto tex = std::make_shared<Texture>();
                    // TODO textures with less than 4 channels... detect automatically?
                    if (stbi_is_hdr(absTexPath.string().c_str()))
                    {
                        tex->loadFromFile(absTexPath, GL_RGBA32F, GL_RGBA, GL_FLOAT, 4);
                    }
                    else if(type == aiTextureType_OPACITY)
                    {
                        tex->loadFromFile(absTexPath, GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1);
                    }
                    else
                    {
                        tex->loadFromFile(absTexPath);
                    }
                    texID = tex->generateHandle();
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
                        if(texID == -1)
                        {
                            hasDiff = 0.0f;
                        }
                        break;
                    case aiTextureType_SPECULAR:
                        gpuMat.specTexture = texID;
                        if (texID == -1)
                        {
                            hasSpec = 0.0f;
                        }
                        break;
                    case aiTextureType_OPACITY:
                        gpuMat.opacityTexture = texID;
                        if (texID == -1)
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

            m_gpuMaterials.push_back(gpuMat);
        }
    }

    m_gpuMaterialBuffer.setStorage(m_gpuMaterials, GL_DYNAMIC_STORAGE_BIT);
    m_gpuMaterialBuffer.bindBase(BufferBindings::Binding::materials);

    m_modelMatrixBuffer.setStorage(m_modelMatrices, GL_DYNAMIC_STORAGE_BIT);
    m_modelMatrixBuffer.bindBase(BufferBindings::Binding::modelMatrices);

    std::cout << "Loading complete: " << filename.string() << std::endl;

}

// OLD CONSTRUCTOR FOR COMPATABILITY
ModelImporter::ModelImporter(const std::experimental::filesystem::path& filename)
    : m_gpuMaterialBuffer(GL_SHADER_STORAGE_BUFFER), m_modelMatrixBuffer(GL_SHADER_STORAGE_BUFFER)
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

void ModelImporter::registerUniforms(ShaderProgram& sp) const
{
    sp.addUniform(m_meshIndexUniform);
    sp.addUniform(m_materialIndexUniform);
}

void ModelImporter::draw(const ShaderProgram& sp) const
{
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

void ModelImporter::drawCulled(const ShaderProgram& sp, Camera& cam, float angle, float ratio, float near, float far) const
{
    glm::vec3 p = cam.getPosition();
    glm::vec3 Z = -glm::normalize(cam.getCenter() - cam.getPosition());

    float tang = glm::tan(angle * 0.5f);
    float nh = near * tang;
    float nw = nh * ratio;
    float fh = far * tang;
    float fw = fh * ratio;

    // X axis of camera with given "up" vector and Z axis
    glm::vec3 X = glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), Z));

    // the real "up" vector is the cross product of Z and X
    glm::vec3 Y = glm::cross(Z, X);

    // compute the centers of the near and far planes
    glm::vec3 nc = p - Z * near;
    glm::vec3 fc = p - Z * far;

    // compute the 4 corners of the frustum on the near plane
    glm::vec3 ntl = nc + Y * nh - X * nw;
    glm::vec3 ntr = nc + Y * nh + X * nw;
    glm::vec3 nbl = nc - Y * nh - X * nw;
    glm::vec3 nbr = nc - Y * nh + X * nw;

    // compute the 4 corners of the frustum on the far plane
    glm::vec3 ftl = fc + Y * fh - X * fw;
    glm::vec3 ftr = fc + Y * fh + X * fw;
    glm::vec3 fbl = fc - Y * fh - X * fw;
    glm::vec3 fbr = fc - Y * fh + X * fw;

    // compute the six planes
    // the function set3Points assumes that the points
    // are given in counter clockwise order
    FrustumGeo f;
    f.set3Points(FrustumGeo::TOP, ntr, ntl, ftl);
    f.set3Points(FrustumGeo::BOTTOM, nbl, nbr, fbr);
    f.set3Points(FrustumGeo::LEFT, ntl, nbl, fbl);
    f.set3Points(FrustumGeo::RIGHT, nbr, ntr, fbr);
    f.set3Points(FrustumGeo::NEAR, ntl, ntr, nbr);
    f.set3Points(FrustumGeo::FAR, ftr, ftl, fbl);

    auto cullFunc = [&f](auto& mesh)
    {
        mesh->setEnabledForRendering(true);
        //for each plane do ...
        for (int i = 0; i < 6; ++i) {
            if (!(f.distance(i, mesh->getBoundingBox()[1]) >= 0.0f ||
                f.distance(i, mesh->getBoundingBox()[0]) >= 0.0f))
            {
                mesh->setEnabledForRendering(false);
            }
        }
    };

    std::for_each(std::execution::par, m_meshes.begin(), m_meshes.end(), cullFunc);

    int i = 0;
    for (const auto& mesh : m_meshes)
    {
        if (mesh->isEnabledForRendering())
        {
            m_meshIndexUniform->setContent(i);
            m_materialIndexUniform->setContent(mesh->getMaterialIndex());
            sp.updateUniforms();
            mesh->draw();
        }
        i++;
    }
}

std::vector<std::shared_ptr<Mesh>> ModelImporter::getMeshes() const
{
    return m_meshes;
}
