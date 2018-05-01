#include "VoxelDebugRenderer.h"
#include "Shader.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "imgui/imgui.h"

VoxelDebugRenderer::VoxelDebugRenderer(const glm::ivec3 gridDim, const ScreenInfo screenInfo)
    : m_gridDim(gridDim), m_screenInfo(screenInfo), m_camera(m_screenInfo.width, m_screenInfo.height, 10.0f),
    m_shaders{ Shader{ "voxeldebug.vert", GL_VERTEX_SHADER }, Shader{ "voxeldebug.geom", GL_GEOMETRY_SHADER }, Shader{ "voxeldebug.frag", GL_FRAGMENT_SHADER } },
    m_sp(m_shaders)
{
    m_numVoxels = gridDim.x * gridDim.y * gridDim.z;
    m_projMat = glm::perspective(glm::radians(60.0f), m_screenInfo.width / static_cast<float>(m_screenInfo.height), m_screenInfo.near, m_screenInfo.far);
    glm::mat4 view = m_camera.getView();
    m_viewUniform = std::make_shared<Uniform<glm::mat4>>("dViewMat", view);
    auto projUniform = std::make_shared<Uniform<glm::mat4>>("dProjMat", m_projMat);
    auto gridDimUniform = std::make_shared<Uniform<glm::ivec3>>("gridDim", gridDim);
    m_voxelSizeUniform = std::make_shared<Uniform<float>>("voxelSize", m_voxelSize);
    m_sp.addUniform(gridDimUniform);
    m_sp.addUniform(m_viewUniform);
    m_sp.addUniform(projUniform);
    m_sp.addUniform(m_voxelSizeUniform);
}

void VoxelDebugRenderer::updateCamera(GLFWwindow* window)
{
    m_camera.update(window);
    m_viewUniform->setContent(m_camera.getView());
}

void VoxelDebugRenderer::draw()
{
    m_sp.showReloadShaderGUI(m_shaders, "DebugRenderer Shaderprogram");


    ImGui::Begin("Voxel Debug Renderer Settings");
    if (ImGui::SliderFloat("Voxel Size", &m_voxelSize, 0.001f, 0.01f))
    {
        m_voxelSizeUniform->setContent(m_voxelSize);
    }
    if (ImGui::Button("Reset Camera"))
    {
        m_camera.reset();
        m_viewUniform->setContent(m_camera.getView());
    }
    ImGui::End();

    m_sp.use();
    m_sp.updateUniforms();

    m_emptyVao.bind();
    glDrawArrays(GL_POINTS, 0, m_numVoxels);
    glBindVertexArray(0);
}