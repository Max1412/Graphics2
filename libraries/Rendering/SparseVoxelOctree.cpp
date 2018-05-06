#include "SparseVoxelOctree.h"

#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

SparseVoxelOctree::SparseVoxelOctree(const std::vector<std::shared_ptr<Mesh>>& scene, const size_t depth)
    : m_N(static_cast<size_t>(glm::pow(2, depth))), m_depth(depth), m_scene(scene)
{

    ////////////////////////////////////////////////////////////////////////////////////
    // Init buffers, textures, atomic counters, etc.

    size_t triCount = 0;
    std::for_each(scene.begin(), scene.end(), [&triCount](auto& mesh) {triCount += mesh->getIndices().size(); });
    triCount /= 3;

    //clear voxel fragment list
    m_voxelFragmentList.setStorage(std::vector<glm::vec4>(triCount * 4, glm::vec4(0.f)), GL_DYNAMIC_STORAGE_BIT);

    //clear voxel color list
    m_voxelFragmentColor.setStorage(std::vector<glm::vec4>(triCount * 4, glm::vec4(0.f)), GL_DYNAMIC_STORAGE_BIT);

    const int powN3 = static_cast<int>(std::pow(m_N, 3));
    //clear node pool
    m_nodePool.setStorage(std::vector<GLint>(powN3, -1), GL_DYNAMIC_STORAGE_BIT);

    //clear node colors
    m_nodeColor.setStorage(std::vector<glm::vec4>(powN3 * 2, glm::vec4(0.f)), GL_DYNAMIC_STORAGE_BIT);

    //clear voxel counter
    m_voxelCounter.setStorage(std::vector<GLuint>{ 0u }, GL_DYNAMIC_STORAGE_BIT);

    //clear node counter
    m_nodeCounter.setStorage(std::vector<GLuint>{ 1u }, GL_DYNAMIC_STORAGE_BIT);

    m_bmin = glm::vec3(std::numeric_limits<float>::max());
    m_bmax = glm::vec3(std::numeric_limits<float>::lowest());
    std::for_each(scene.begin(), scene.end(), [&](auto& mesh)
    {
        glm::mat2x3 bbox = mesh->getBoundingBox();
        m_bmin = glm::min(m_bmin, bbox[0]);
        m_bmax = glm::max(m_bmax, bbox[1]);
    });

    // make cubic
    const glm::vec3 c = (m_bmin + m_bmax) / 2.0f;
    const float halfSize = glm::compMax(m_bmax-m_bmin) / 2.0f;
    m_bmin = c - halfSize;
    m_bmax = c + halfSize;

    ////////////////////////////////////////////////////////////////////////////////////
    // Init shader uniforms
    const auto u_proj = std::make_shared<Uniform<glm::mat4>>("projectionMatrix", glm::ortho(
        m_bmin.x, m_bmax.x,
        m_bmin.y, m_bmax.y,
        -m_bmax.z, -m_bmin.z
    ));
    m_voxelGenShader.addUniform(u_proj);

    const auto u_res = std::make_shared<Uniform<glm::uvec3>>("res", glm::uvec3(m_N, m_N, m_N));
    m_voxelGenShader.addUniform(u_res);

    m_startIndexUniform = std::make_shared<Uniform<int>>("startIndex", 0);
    m_nodeCreationShader.addUniform(m_startIndexUniform);
    m_mipMapShader.addUniform(m_startIndexUniform);

    m_modelMatrixUniform = std::make_shared<Uniform<glm::mat4>>("modelMatrix", glm::mat4(1.0f));
    m_voxelGenShader.addUniform(m_modelMatrixUniform);

    ///////////////////////////////////////////////////////////////////////////////////
    update();
    ///////////////////////////////////////////////////////////////////////////////////
}

void SparseVoxelOctree::bind() const
{
    m_nodePool.bindBase(2);
    m_nodeColor.bindBase(3);  
}

void SparseVoxelOctree::update()
{
    double t1 = 0.f;
    GLuint voxelCount, nodeCount;

    ////////////////////////////////////////////////////////////////////////////////////
    // clear and bind buffers, textures, atomic counters, etc.

    if constexpr(util::debugmode) { glFinish(); t1 = glfwGetTime(); }

    //bind Buffers
    m_voxelFragmentList.bindBase(0);
    m_voxelFragmentColor.bindBase(1);
    m_nodePool.bindBase(2);
    m_nodeColor.bindBase(3);
    m_voxelCounter.bindBase(0);
    m_nodeCounter.bindBase(1);

    //clear SSBOs
    glGetNamedBufferSubData(m_voxelCounter.getHandle(), 0, 4, &voxelCount);
    glGetNamedBufferSubData(m_nodeCounter.getHandle(), 0, 4, &nodeCount);

    glm::vec4 zero_vec = glm::vec4(0.f); 
    const GLint clear_val = -1;
    glClearNamedBufferSubData(m_voxelFragmentList.getHandle(), GL_RGBA32F, 0, (voxelCount+1) * 16 /*4 * 4*/, GL_RGBA, GL_FLOAT, glm::value_ptr(zero_vec));
    glClearNamedBufferSubData(m_voxelFragmentColor.getHandle(), GL_RGBA32F, 0, (voxelCount+1) * 16 /*4 * 4*/, GL_RGBA, GL_FLOAT, glm::value_ptr(zero_vec));
    glClearNamedBufferSubData(m_nodeColor.getHandle(), GL_RGBA32F, 0, (nodeCount+1) * 128 /*8 * 4 * 4*/, GL_RGBA, GL_FLOAT, glm::value_ptr(zero_vec));
    glClearNamedBufferSubData(m_nodePool.getHandle(), GL_R32I, 0, (nodeCount+1) * 32 /*8 * 4*/, GL_RED_INTEGER, GL_INT, &clear_val);

    //clear atomic counters
    GLuint clear_val_atomic = 0u;
    glClearNamedBufferSubData(m_voxelCounter.getHandle(), GL_R32UI, 0, 4, GL_RED_INTEGER, GL_UNSIGNED_INT, &clear_val_atomic);
    clear_val_atomic = 1u;
    glClearNamedBufferSubData(m_nodeCounter.getHandle(), GL_R32UI, 0, 4, GL_RED_INTEGER, GL_UNSIGNED_INT, &clear_val_atomic);

    //set octree root value
    const GLint root_val = 8;
    m_nodePool.setContentSubData(root_val, 0);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

    if constexpr(util::debugmode) { glFinish(); std::cout << "Buffer clearing took: " << (glfwGetTime() - t1) * 1000 << "ms \n"; }

    ////////////////////////////////////////////////////////////////////////////////////
    // Voxelization into uniform grid, i.e. creation of voxel fragment list

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
    const float Nf = static_cast<float>(m_N);
    glViewportIndexedf(1, 0.0f, 0.0f, Nf, Nf);
    glViewportIndexedf(2, 0.0f, 0.0f, Nf, Nf);
    glViewportIndexedf(3, 0.0f, 0.0f, Nf, Nf);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    m_voxelGenShader.use();
    if constexpr(util::debugmode) { glFinish(); t1 = glfwGetTime(); }
    std::for_each(m_scene.begin(), m_scene.end(), [&](auto& mesh)
    {
        m_modelMatrixUniform->setContent(mesh->getModelMatrix());
        m_voxelGenShader.updateUniforms();
        mesh->draw();
    });
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
    if constexpr(util::debugmode) {glFinish(); std::cout << "Voxelization took: " << (glfwGetTime() - t1) * 1000 << "ms \n";}

    glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);

    glGetNamedBufferSubData(m_voxelCounter.getHandle(), 0, 4, &voxelCount);
    if constexpr(util::debugmode) std::cout << voxelCount << " filled Voxels \n";

    ///////////////////////////////////////////////////////////////////////////////////
    // Octree generation

    std::vector<int> levelStartIndices;
    levelStartIndices.push_back(8);
    if constexpr(util::debugmode) { glFinish(); t1 = glfwGetTime(); }
    for (int i = 1; i < m_depth; ++i) {
        m_flagShader.use();
        glDispatchCompute(static_cast<GLuint>(glm::ceil(voxelCount / 64.f)), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glGetNamedBufferSubData(m_nodeCounter.getHandle(), 0, 4, &nodeCount);
        levelStartIndices.push_back(int((nodeCount + 1) * 8));
        m_startIndexUniform->setContent(levelStartIndices[i - 1]);
        m_nodeCreationShader.use();
        glDispatchCompute(static_cast<GLuint>(glm::pow(8, i - 1)), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
    }
    glGetNamedBufferSubData(m_nodeCounter.getHandle(), 0, 4, &nodeCount);
    levelStartIndices.push_back(int((nodeCount + 1) * 8));
    if constexpr(util::debugmode) {
        glFinish();
        std::cout << "Tree building took: " << (glfwGetTime() - t1) * 1000 << "ms \n";

        std::cout << nodeCount * 8 << " Nodes \n";
    }

    ///////////////////////////////////////////////////////////////////////////////////
    // Initialization of leaf node data

    if constexpr(util::debugmode) { glFinish(); t1 = glfwGetTime(); }
    m_leafInitShader.use();
    glDispatchCompute(static_cast<GLuint>(glm::ceil(voxelCount / 64.f)), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if constexpr(util::debugmode) { glFinish(); std::cout << "Leaf initialization took: " << (glfwGetTime() - t1) * 1000 << "ms \n"; }

    ///////////////////////////////////////////////////////////////////////////////////
    // Mipmap values

    if constexpr(util::debugmode) { glFinish(); t1 = glfwGetTime(); }
    m_mipMapShader.use();
    for (int64_t i = m_depth - 2; i >= 0; --i) {
        m_startIndexUniform->setContent(levelStartIndices[i]);
        m_mipMapShader.updateUniforms();
        glDispatchCompute((levelStartIndices[i + 1] - levelStartIndices[i]) / 8, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    m_startIndexUniform->setContent(0);
    m_mipMapShader.updateUniforms();
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if constexpr(util::debugmode) {
        glFinish();
        std::cout << "Mipmapping values took: " << (glfwGetTime() - t1) * 1000 << "ms \n";

        glm::vec4 rootNodeColor;
        glGetNamedBufferSubData(m_nodeColor.getHandle(), 0, 4 * 4, &rootNodeColor);
        std::cout << "Root color: (" << rootNodeColor.x << ", " << rootNodeColor.y << ", " << rootNodeColor.z << ", " << rootNodeColor.w << ") \n";
    }

    ///////////////////////////////////////////////////////////////////////////////////

}

glm::vec3 SparseVoxelOctree::getBMin() const
{
    return m_bmin;
}

glm::vec3 SparseVoxelOctree::getBMax() const
{
    return m_bmax;
}
