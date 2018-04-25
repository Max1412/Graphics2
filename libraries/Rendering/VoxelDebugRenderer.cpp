#include "VoxelDebugRenderer.h"
#include "Shader.h"
#include "ShaderProgram.h"
#include "SimpleTrackball.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

VoxelDebugRenderer::VoxelDebugRenderer(const glm::ivec3 gridDim, const ScreenInfo screenInfo)
    : m_gridDim(gridDim), m_screenInfo(screenInfo), m_camera(m_screenInfo.width, m_screenInfo.height, 1.0f),
    m_shaders{ Shader{ "voxeldebug.vs", GL_VERTEX_SHADER }, Shader{ "voxeldebug.gs", GL_GEOMETRY_SHADER }, Shader{ "voxeldebug.fs", GL_FRAGMENT_SHADER } },
    m_sp(m_shaders)
{
    m_numVoxels = gridDim.x * gridDim.y * gridDim.z;
    m_sp.use();
    m_projMat = glm::perspective(glm::radians(60.0f), m_screenInfo.width / static_cast<float>(m_screenInfo.height), m_screenInfo.near, m_screenInfo.far);
    m_viewUniform = std::make_shared<Uniform<glm::mat4>>("debugViewMat", m_camera.getView());
    auto projUniform = std::make_shared<Uniform<glm::mat4>>("debugProjMat", m_projMat);
    auto gridDimUniform = std::make_shared<Uniform<glm::ivec3>>("gridDim", gridDim);
    m_sp.addUniform(projUniform);
    m_sp.addUniform(m_viewUniform);

}

void VoxelDebugRenderer::draw(GLFWwindow* window)
{
    m_camera.update(window);
    m_viewUniform->setContent(m_camera.getView());
    m_sp.use();
    m_sp.updateUniforms();
    glDrawArrays(GL_POINTS, 0, m_numVoxels);
}