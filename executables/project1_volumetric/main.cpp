#include <glbinding/gl/gl.h>
using namespace gl;

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/Uniform.h"
#include "Rendering/Image.h"
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/Camera.h"
#include "Rendering/VoxelDebugRenderer.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/FrameBuffer.h"

constexpr int screenWidth = 1600;
constexpr int screenHeight = 900;
constexpr float screenNear = 0.1f;
constexpr int screenFar = 1000;
constexpr int gridWidth = screenWidth / 10;
constexpr int gridHeight = screenHeight / 10;
constexpr int gridDepth = screenFar / 10;
constexpr int groupSize = 4;

constexpr bool renderimgui = true;

struct PlayerCameraInfo
{
    glm::mat4 playerViewMatrix;
    glm::mat4 playerProjMatrix;
    glm::vec3 playerCameraPosition;
    float pad = 0.0f;
    float near = screenNear;
};

int main()
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(screenWidth, screenHeight, "Rapid Testing Executable");
    glfwSwapInterval(0);
    // init opengl
    util::initGL();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    Image voxelGrid(GL_TEXTURE_3D, GL_NEAREST, GL_NEAREST);
    voxelGrid.initWithoutData3D(gridWidth, gridHeight, gridDepth, GL_RGBA32F);
    GLuint64 handle = voxelGrid.generateImageHandle(GL_RGBA32F);
    voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

    Buffer imageHoldingSSBO(GL_SHADER_STORAGE_BUFFER);
    imageHoldingSSBO.setStorage(std::vector<GLuint64>{ handle }, GL_DYNAMIC_STORAGE_BIT);
    imageHoldingSSBO.bindBase(0);

    Shader perVoxelShader("perVoxel3.comp", GL_COMPUTE_SHADER);
    ShaderProgram sp({ perVoxelShader });

    Shader accumShader("accumulateVoxels.comp", GL_COMPUTE_SHADER);
    ShaderProgram accumSp({ accumShader });

    auto u_gridDim = std::make_shared<Uniform<glm::ivec3>>("gridDim", glm::ivec3(gridWidth, gridHeight, gridDepth));
    //sp.addUniform(u_gridDim);
    accumSp.addUniform(u_gridDim);

    Camera playerCamera(screenWidth, screenHeight, 10.0f);
    glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), screenWidth / static_cast<float>(screenHeight), screenNear, static_cast<float>(screenFar));

    Buffer matrixSSBO(GL_SHADER_STORAGE_BUFFER);
    matrixSSBO.setStorage(std::array<PlayerCameraInfo, 1>{ {playerCamera.getView(), playerProj, playerCamera.getPosition(), 0.0f, screenNear }}, GL_DYNAMIC_STORAGE_BIT);
    matrixSSBO.bindBase(1);

    VoxelDebugRenderer vdbgr({ gridWidth, gridHeight, gridDepth }, { screenWidth, screenHeight, screenNear, static_cast<float>(screenFar) });

    Timer timer;

    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window))
    {
        timer.start();

        glfwPollEvents();

        if constexpr (renderimgui)
            ImGui_ImplGlfwGL3_NewFrame();
         
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if constexpr (renderimgui)
        {
            sp.showReloadShaderGUI({ perVoxelShader }, "Voxel");
            //accumSp.showReloadShaderGUI({ accumShader }, "Accumulation");
        }

        voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

        sp.use();
        glDispatchCompute(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(groupSize))),
            static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(groupSize))),
            static_cast<GLint>(std::ceil(gridDepth / static_cast<float>(groupSize))));
		//glDispatchComputeGroupSizeARB(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(groupSize))),
        //static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(groupSize))),
        //static_cast<GLint>(std::ceil(gridDepth / static_cast<float>(groupSize))), groupSize, groupSize, groupSize);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        //accumSp.use();
        //glDispatchCompute(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(8))),
        //    static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(8))), 1);

        //glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        vdbgr.draw(window);

        timer.stop();

        if constexpr (renderimgui)
        {
            timer.drawGuiWindow(window);
            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(window);
    }

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
