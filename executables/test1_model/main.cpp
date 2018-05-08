#include <glbinding/gl/gl.h>
#include "Rendering/Binding.h"
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
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/Pilotview.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/SkyBoxCube.h"
#include "Rendering/Cubemap.h"
#include "Rendering/Quad.h"
#include "Rendering/FrameBuffer.h"

constexpr int width = 1600;
constexpr int height = 900;

int main()
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Test: Model Loading");
    glfwSwapInterval(0);

    // init opengl
    util::initGL();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    Pilotview playerCamera(width, height);
    glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 10000.0f);
    auto projUniform = std::make_shared<Uniform<glm::mat4>>("projectionMatrix", playerProj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("viewMatrix", playerCamera.getView());

    ModelImporter modelLoader("sponza/sponza.obj", 1);

    Shader modelVertexShader("modelVert.vert", GL_VERTEX_SHADER, BufferBindings::g_definitions);
    Shader modelFragmentShader("modelFrag.frag", GL_FRAGMENT_SHADER, BufferBindings::g_definitions);
    ShaderProgram sp(modelVertexShader, modelFragmentShader);
    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);

    modelLoader.registerUniforms(sp);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);

    Timer timer;

    // render loop
    while (!glfwWindowShouldClose(window))
    {
        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();
        
        playerCamera.update(window);
        viewUniform->setContent(playerCamera.getView());
        sp.updateUniforms();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        sp.use();

        // DRAW
        //modelLoader.draw(sp);
        modelLoader.drawCulled(sp, playerCamera, glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 10000.0f);

        timer.stop();
        timer.drawGuiWindow(window);

        sp.showReloadShaderGUI({modelVertexShader, modelFragmentShader}, "Forward shader");

        ImGui::Render();
        ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
