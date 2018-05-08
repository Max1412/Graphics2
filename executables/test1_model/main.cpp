#include <glbinding/gl/gl.h>
#include "Rendering/Binding.h"
#include <execution>
#include "Rendering/Light.h"
using namespace gl;

#include <GLFW/glfw3.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Uniform.h"
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/Pilotview.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

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
    auto cameraPosUniform = std::make_shared<Uniform<glm::vec3>>("cameraPos", playerCamera.getPosition());


    Shader modelVertexShader("modelVert.vert", GL_VERTEX_SHADER, BufferBindings::g_definitions);
    Shader modelFragmentShader("modelFrag.frag", GL_FRAGMENT_SHADER, BufferBindings::g_definitions);
    ShaderProgram sp(modelVertexShader, modelFragmentShader);
    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(cameraPosUniform);

    ModelImporter modelLoader("sponza/sponza.obj", 1);
    modelLoader.registerUniforms(sp);

    // "generate" lights
    LightManager lightMngr;
    for (int i = 0; i < 1; i++)
    {
        glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec4 pos = glm::vec4(0.0f, 200.0f, 0.0f, 1.0f);
        glm::vec3 dir = glm::normalize(glm::vec3(0.0f) - glm::vec3(pos));
        float constant = 1.0f;
        float linear = 0.09f;
        float quadratic = 0.032f;
        float cutOff = glm::cos(glm::radians(12.5f));
        float outerCutOff = glm::cos(glm::radians(15.0f));
        auto li = std::make_shared<Light>(color, pos, dir, constant, linear, quadratic, cutOff, outerCutOff, glm::ivec2(1600, 900));

        lightMngr.addLight(li);
    }
    lightMngr.uploadLightsToGPU();

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);

    Timer timer;

    bool cullingOn = true;

    // render loop
    while (!glfwWindowShouldClose(window))
    {
        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();
        
        playerCamera.update(window);
        viewUniform->setContent(playerCamera.getView());
        cameraPosUniform->setContent(playerCamera.getPosition());
        sp.updateUniforms();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        sp.use();

        // DRAW
        ImGui::Checkbox("Draw with View Frustum Culling", &cullingOn);
        if(cullingOn)
            modelLoader.drawCulled(sp, playerCamera, glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 10000.0f);
        else
        {
            //std::for_each(std::execution::par, modelLoader.getMeshes().begin(), modelLoader.getMeshes().end(), [](auto &Mesh) { Mesh->setEnabledForRendering(true); });
            modelLoader.draw(sp);
            
        }
        lightMngr.renderShadowMaps(modelLoader.getMeshes());

        lightMngr.showLightGUIs();


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
