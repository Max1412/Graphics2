#include <glbinding/gl/gl.h>
#include "Rendering/LightManager.h"
using namespace gl;

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <memory>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/Uniform.h"
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/Trackball.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/Quad.h"
#include "Rendering/FrameBuffer.h"

const unsigned int width = 1600;
const unsigned int height = 900;

struct MaterialInfo
{
    glm::vec3 diffColor;
    float kd;
    glm::vec3 specColor;
    float ks;
    float shininess;
    float kt;
    int reflective;
    float pad2 = 0.0f;
};

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

    // FBO stuff
    const Shader fboVS("texSFQ.vert", GL_VERTEX_SHADER);
    const Shader fboFS("fbo.frag", GL_FRAGMENT_SHADER);
    ShaderProgram fboSP(fboVS, fboFS);
    fboSP.use();

    Quad fboQuad;

    auto fboTex = std::make_shared<Texture>();;
    fboTex->initWithoutData(width, height, GL_RGBA8);
    fboTex->generateHandle();

    // put the texture handle into a SSBO
    const auto fboTexHandle = fboTex->getHandle();
    Buffer fboTexHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    fboTexHandleBuffer.setStorage(std::array<GLuint64, 1>{fboTexHandle}, GL_DYNAMIC_STORAGE_BIT);
    fboTexHandleBuffer.bindBase(static_cast<BufferBindings::Binding>(6));

    FrameBuffer fbo({ fboTex });

    // actual stuff
    const Shader vs("shadowmapping.vert", GL_VERTEX_SHADER);
    const Shader fs("shadowmapping.frag", GL_FRAGMENT_SHADER, BufferBindings::g_definitions);
    ShaderProgram sp(vs, fs);

    auto meshes = ModelImporter::loadAllMeshesFromFile("bunny.obj");
    auto bunny = meshes.at(0);

    // create a plane
    std::vector<glm::vec3> planePositions =
    {
        glm::vec3(-1, 0, -1), glm::vec3(-1, 0, 1), glm::vec3(1, 0, 1), glm::vec3(1, 0, -1)
    };

    std::vector<glm::vec3> planeNormals =
    {
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)
    };

    std::vector<unsigned> planeIndices =
    {
        0, 1, 2,
        2, 3, 0
    };

    std::shared_ptr<Mesh> plane = std::make_shared<Mesh>(planePositions, planeNormals, planeIndices);

    sp.use();

    // create matrices for uniforms
    Trackball camera(width, height, 10.0f);
    glm::mat4 view = camera.getView();

    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 1000.0f);
    glm::mat4 model(1.0f);
    model = translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
    //model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));
    bunny->setModelMatrix(model);

    glm::mat4 PlaneModel(1.0f);
    PlaneModel = translate(PlaneModel, glm::vec3(0.0f, -1.34f, 0.0f));
    PlaneModel = scale(PlaneModel, glm::vec3(5.0f));
    plane->setModelMatrix(PlaneModel);

    // create matrix uniforms and add them to the shader program
    const auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
    auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);
    auto camPosUniform = std::make_shared<Uniform<glm::vec3>>("camPos", camera.getPosition());

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(modelUniform);
    sp.addUniform(camPosUniform);

    glm::vec3 ambient(0.5f);
    const auto ambientLightUniform = std::make_shared<Uniform<glm::vec3>>("lightAmbient", ambient);
    sp.addUniform(ambientLightUniform);

    // "generate" lights
    LightManager lightMngr;

    float cutOff = glm::cos(glm::radians(30.0f));
    float outerCutOff = glm::cos(glm::radians(35.0f));
    auto li = std::make_shared<Light>(glm::vec3(1.0f), glm::vec3(5.0f), glm::vec3(-1.0f), 0.05f, 0.002f, 0.0f, cutOff, outerCutOff, 30.0f);

    lightMngr.addLight(li);
    lightMngr.uploadLightsToGPU();

    // set up materials
    std::vector<MaterialInfo> mvec;
    MaterialInfo m;
    m.diffColor = glm::vec3(0.9f);
    m.kd = 0.5f;
    m.specColor = glm::vec3(0.9f);
    m.ks = 3.0f;
    m.shininess = 100.0f;
    m.kt = 0.0f;
    m.reflective = 0;
    mvec.push_back(m);

    MaterialInfo m2;
    m2.diffColor = glm::vec3(0.9f);
    m2.kd = 0.3f;
    m2.specColor = glm::vec3(0.9f);
    m2.ks = 0.3f;
    m2.shininess = 20.0f;
    m2.kt = 0.0f;
    m2.reflective = 0;
    mvec.push_back(m2);

    // first material is for bunny, second for plane
    bunny->setMaterialID(0);
    plane->setMaterialID(1);

    Buffer materialBuffer(GL_SHADER_STORAGE_BUFFER);
    materialBuffer.setStorage(mvec, GL_DYNAMIC_STORAGE_BIT);
    materialBuffer.bindBase(BufferBindings::Binding::materials);

    const glm::vec4 clearColor(0.1f);

    // shading uniforms
    auto MaterialIDUniform = std::make_shared<Uniform<int>>("matIndex", bunny->getMaterialID());
    sp.addUniform(MaterialIDUniform);

    // shadow map displaying stuff //////////
    // FBO stuff
    const Shader smfboFS("smfbo.frag", GL_FRAGMENT_SHADER);
    ShaderProgram smfboSP(fboVS, smfboFS);
    smfboSP.use();

    // put the sm texture handle into a SSBO
    const auto smfboTexHandle = lightMngr.getLights()[0]->getGpuLight().shadowMap;
    Buffer smfboTexHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    smfboTexHandleBuffer.setStorage(std::array<GLuint64, 1>{smfboTexHandle}, GL_DYNAMIC_STORAGE_BIT);
    smfboTexHandleBuffer.bindBase(static_cast<BufferBindings::Binding>(7));
    bool displayShadowMap = false;

    // regular stuff
    Timer timer;

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);

    const float deltaAngle = 0.1f;
    bool rotate = true;

    bool useFBO = true;

    // render loop
    while (!glfwWindowShouldClose(window))
    {
        timer.start();
        if (rotate)
        {
            const glm::mat4 newModel = glm::rotate(bunny->getModelMatrix(), glm::radians(deltaAngle), glm::vec3(0.0f, 1.0f, 0.0f));
            bunny->setModelMatrix(newModel);
        }

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();
        sp.showReloadShaderGUI({ vs, fs }, "Forward Lighting");
        fboSP.showReloadShaderGUI({ fboVS, fboFS }, "FBO");

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("FBO settings");
            ImGui::Checkbox("Render to FBO", &useFBO);
            ImGui::Checkbox("Display Shadow Map", &displayShadowMap);
            ImGui::Checkbox("Rotate Model", &rotate);
            ImGui::End();
        }

        lightMngr.showLightGUIs();

        // shadow mapping pass ///////
        {
            lightMngr.renderShadowMaps({bunny, plane});
        }
        // end shadow mapping pass /////////////

        //ImGui::ShowTestWindow();
        if (useFBO)
            fbo.bind(); // render into fbo

        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.update(window);

        viewUniform->setContent(camera.getView());
        camPosUniform->setContent(camera.getPosition());
        sp.use();

        // prepare first mesh (bunny)
        modelUniform->setContent(bunny->getModelMatrix());
        MaterialIDUniform->setContent(bunny->getMaterialID());
        sp.updateUniforms();

        bunny->draw();

        // prepare plane
        modelUniform->setContent(plane->getModelMatrix());
        MaterialIDUniform->setContent(plane->getMaterialID());
        sp.updateUniforms();

        plane->draw();

        if (useFBO)
        {
            fbo.unbind(); // render to screen now
            glDisable(GL_DEPTH_TEST);
            glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            fboSP.use();
            fboQuad.draw();
            glEnable(GL_DEPTH_TEST);
        }

        if (displayShadowMap)
        {
            fbo.unbind(); // render to screen now
            glDisable(GL_DEPTH_TEST);
            glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            smfboSP.use();
            fboQuad.draw();
            glEnable(GL_DEPTH_TEST);
        }

        timer.stop();
        timer.drawGuiWindow(window);

        ImGui::Render();
        ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
