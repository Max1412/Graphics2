#pragma once

#include <vector>
#include "Uniform.h"
#include "ShaderProgram.h"
#include "glm/glm.hpp"
#include "VertexArray.h"
#include "Trackball.h"
#include "Pilotview.h"

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
    void draw() const;
	void drawGuiContent();
	void drawCameraGuiContent();

private:
    glm::ivec3 m_gridDim;
    ScreenInfo m_screenInfo;
    Pilotview m_camera;
    std::vector<Shader> m_shaders;
    ShaderProgram m_sp;
    int m_numVoxels;
    float m_voxelSize = 0.004f;
    std::shared_ptr<Uniform<float>> m_voxelSizeUniform;
    std::shared_ptr<Uniform<int>> m_positionSourceUniform;
    std::shared_ptr<Uniform<int>> m_dataModeUniform;
    glm::mat4 m_projMat;
    std::shared_ptr<Uniform<glm::mat4>> m_viewUniform;
    bool m_wireframe = false;
    VertexArray m_emptyVao;
};