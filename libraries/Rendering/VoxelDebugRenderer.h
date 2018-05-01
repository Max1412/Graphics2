#pragma once

#include <vector>
#include "Uniform.h"
#include "Camera.h"
#include "ShaderProgram.h"
#include "glm/glm.hpp"
#include "VertexArray.h"

struct ScreenInfo
{
    int width;
    int height;
    float near;
    float far;
};

class Shader;
class Camera;
class VoxelDebugRenderer
{
public:
    VoxelDebugRenderer(const glm::ivec3 gridDim, const ScreenInfo screenInfo);
    void updateCamera(GLFWwindow* window);
    void draw();

private:
    glm::ivec3 m_gridDim;
    ScreenInfo m_screenInfo;
    Camera m_camera;
    std::vector<Shader> m_shaders;
    ShaderProgram m_sp;
    int m_numVoxels;
    float m_voxelSize = 0.004f;
    std::shared_ptr<Uniform<float>> m_voxelSizeUniform;
    glm::mat4 m_projMat;
    std::shared_ptr<Uniform<glm::mat4>> m_viewUniform;

    VertexArray m_emptyVao;
};