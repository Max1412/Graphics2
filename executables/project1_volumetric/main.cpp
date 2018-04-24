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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/Quad.h"
#include "Rendering/FrameBuffer.h"

const unsigned int width = 160;
const unsigned int height = 90;
const unsigned int depth = 100;
const int groupSize = 8;

int main()
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Rapid Testing Executable");
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
	voxelGrid.initWithoutData3D(width, height, depth, GL_RGBA32F);
	GLuint64 handle = voxelGrid.generateImageHandle(GL_RGBA32F);
	
	Buffer imageHoldingSSBO(GL_SHADER_STORAGE_BUFFER);
	imageHoldingSSBO.setStorage(std::vector<GLuint64>{ handle }, GL_DYNAMIC_STORAGE_BIT);
	imageHoldingSSBO.bindBase(0);

	Shader perVoxelShader("perVoxel3.comp", GL_COMPUTE_SHADER);
	ShaderProgram sp({ perVoxelShader });
	sp.use();

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		//glDispatchCompute(width / groupSize, height / groupSize, depth / groupSize);
		glDispatchComputeGroupSizeARB(width / groupSize, height / groupSize, depth / groupSize, groupSize, groupSize, groupSize);

		glfwSwapBuffers(window);
	}

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
