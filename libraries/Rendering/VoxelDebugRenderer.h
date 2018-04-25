#pragma once

#include <vector>
#include "Uniform.h"
#include "SimpleTrackball.h"
#include "ShaderProgram.h"
#include "glm/glm.hpp"

struct ScreenInfo
{
    int width;
    int height;
    float near;
    float far;
};

class Shader;
class SimpleTrackball;
class VoxelDebugRenderer
{
public:
    VoxelDebugRenderer(const glm::ivec3 gridDim, const ScreenInfo screenInfo);
    void draw(GLFWwindow* window);

private:
    glm::ivec3 m_gridDim;
    ScreenInfo m_screenInfo;
    SimpleTrackball m_camera;
    std::vector<Shader> m_shaders;
    ShaderProgram m_sp;
    int m_numVoxels;
    glm::mat4 m_projMat;
    std::shared_ptr<Uniform<glm::mat4>> m_viewUniform;
};