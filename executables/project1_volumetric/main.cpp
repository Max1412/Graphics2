#include <glbinding/gl/gl.h>
using namespace gl;

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <memory>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/Uniform.h"
#include "Rendering/Image.h"
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/SimpleTrackball.h"
#include "Rendering/VoxelDebugRenderer.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/Quad.h"
#include "Rendering/FrameBuffer.h"

constexpr int screenWidth = 1600;
constexpr int screenHeight = 900;
constexpr int screenFar = 1000;
constexpr int gridWidth = screenWidth / 10;
constexpr int gridHeight = screenHeight / 10;
constexpr int gridDepth = screenFar / 10;
constexpr int groupSize = 8;

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

    // get list of OpenGL extensions (can be searched later if needed)
    std::vector<std::string> extensions = util::getGLExtenstions();

	Image voxelGrid(GL_TEXTURE_3D, GL_NEAREST, GL_NEAREST);
	voxelGrid.initWithoutData3D(gridWidth, gridHeight, gridDepth, GL_RGBA32F);
	GLuint64 handle = voxelGrid.generateImageHandle(GL_RGBA32F);
	
	Buffer imageHoldingSSBO(GL_SHADER_STORAGE_BUFFER);
	imageHoldingSSBO.setStorage(std::vector<GLuint64>{ handle }, GL_DYNAMIC_STORAGE_BIT);
	imageHoldingSSBO.bindBase(0);

	Shader perVoxelShader("perVoxel3.comp", GL_COMPUTE_SHADER);
	ShaderProgram sp({ perVoxelShader });
	sp.use();

    SimpleTrackball playerCamera(screenWidth, screenHeight, 1.0f);
    glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), screenWidth / static_cast<float>(screenHeight), 0.1f, static_cast<float>(screenFar));

    Buffer matrixSSBO(GL_SHADER_STORAGE_BUFFER);
    matrixSSBO.setStorage(std::array<glm::mat4, 2>{ playerCamera.getView(), playerProj }, GL_DYNAMIC_STORAGE_BIT);
    matrixSSBO.bindBase(1);

    VoxelDebugRenderer vdbgr({ gridWidth, gridHeight, gridDepth }, { screenWidth, screenHeight, 0.1f, static_cast<float>(screenFar) });

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		//glDispatchCompute(gridWidth / groupSize, groupSize / groupSize, groupSize / groupSize);
        sp.use();
        sp.updateUniforms();
		glDispatchComputeGroupSizeARB(gridWidth / groupSize, gridHeight / groupSize, gridDepth / groupSize, groupSize, groupSize, groupSize);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        vdbgr.draw(window);

		glfwSwapBuffers(window);
	}

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
