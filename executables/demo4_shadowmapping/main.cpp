#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
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
#include "Rendering/SimpleTrackball.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/Quad.h"
#include "Rendering/FrameBuffer.h"

const unsigned int width = 1600;
const unsigned int height = 900;

struct LightInfo
{
    glm::vec4 pos; //pos.w=0 dir., pos.w=1 point light
    glm::vec3 col;
    float spot_cutoff; //no spotlight if cutoff=0
    glm::vec3 spot_direction;
    float spot_exponent;
};

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
    // init glbinding
    glbinding::Binding::initialize();

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

    Texture fboTex;
    fboTex.initWithoutData(width, height, GL_RGBA8);
    fboTex.generateHandle();

    // put the texture handle into a SSBO
    const auto fboTexHandle = fboTex.getHandle();
    Buffer fboTexHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    fboTexHandleBuffer.setStorage(std::array<GLuint64, 1>{fboTexHandle}, GL_DYNAMIC_STORAGE_BIT);
    fboTexHandleBuffer.bindBase(6);

    FrameBuffer fbo({fboTex});

    // actual stuff
    const Shader vs("shadowmapping.vert", GL_VERTEX_SHADER);
    const Shader fs("shadowmapping.frag", GL_FRAGMENT_SHADER);
    ShaderProgram sp(vs, fs);

    ModelImporter mi("bunny.obj");
    auto meshes = mi.getMeshes();
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

    Mesh plane(planePositions, planeNormals, planeIndices);

    sp.use();

    // create matrices for uniforms
    SimpleTrackball camera(width, height, 10.0f);
    glm::mat4 view = camera.getView();

    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 1000.0f);
    glm::mat4 model(1.0f);
    model = translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
    //model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));
    bunny->setModelMatrix(model);

    glm::mat4 PlaneModel(1.0f);
    PlaneModel = translate(PlaneModel, glm::vec3(0.0f, -1.34f, 0.0f));
    PlaneModel = scale(PlaneModel, glm::vec3(5.0f));
    plane.setModelMatrix(PlaneModel);

    // create matrix uniforms and add them to the shader program
    const auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
    auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(modelUniform);

    glm::vec3 ambient(0.5f);
    const auto ambientLightUniform = std::make_shared<Uniform<glm::vec3>>("lightAmbient", ambient);
    sp.addUniform(ambientLightUniform);

    // "generate" lights
    std::vector<LightInfo> lvec;
    for (int i = 0; i < 1; i++)
    {
        LightInfo li;
        const glm::mat4 rotMat = rotate(glm::mat4(1.0f), glm::radians(i * (360.0f / 5.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
        li.pos = rotMat * (glm::vec4(i * 3.0f, i * 3.0f, i * 3.0f, 1.0f) + glm::vec4(5.0f, 5.0f, 5.0f, 0.0f));
        li.col = normalize(glm::vec3((i) % 5, (i + 1) % 5, (i + 2) % 5));
        if (i % 2)
        {
            li.col = normalize(glm::vec3((i - 1) % 5, (i) % 5, (i + 1) % 5));
            li.col = normalize(glm::vec3(1.0f) - li.col);
        }
        std::cout << to_string(li.col) << std::endl;
        if (i == 3)
        {
            li.spot_cutoff = 0.1f;
        }
        else
        {
            li.spot_cutoff = 0.0f;
        }
        li.spot_direction = normalize(glm::vec3(0.0f) - glm::vec3(li.pos));
        li.spot_exponent = 1.0f;

        lvec.push_back(li);
    }

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
    plane.setMaterialID(1);

    // create buffers for materials and lights
    Buffer lightBuffer(GL_SHADER_STORAGE_BUFFER);
    lightBuffer.setStorage(lvec, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
    lightBuffer.bindBase(0);

    Buffer materialBuffer(GL_SHADER_STORAGE_BUFFER);
    materialBuffer.setStorage(mvec, GL_DYNAMIC_STORAGE_BIT);
    materialBuffer.bindBase(1);

    const glm::vec4 clearColor(0.1f);

    // shading uniforms
    auto MaterialIDUniform = std::make_shared<Uniform<int>>("matIndex", bunny->getMaterialID());

    sp.addUniform(MaterialIDUniform);

    // Shadow Mapping stuff ////////////////////////////////////////////////////
    const int shadowWidth = width, shadowHeight = height;
    Texture shadowTexture(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST);
    shadowTexture.initWithoutData(shadowWidth, shadowHeight, GL_DEPTH_COMPONENT32F);
    shadowTexture.setWrap(GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
    shadowTexture.generateHandle();

    FrameBuffer shadowMapFBO(GL_DEPTH_ATTACHMENT, {shadowTexture});

    const float nearPlane = 3.0f, farPlane = 18.0f;
    const glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, nearPlane, farPlane);

    glm::mat4 lightView = lookAt(glm::vec3(lvec.at(0).pos),
                                 glm::vec3(0.0f), // aimed at the center
                                 glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 lightSpaceMatrix = lightProjection * lightView;
    auto lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", lightSpaceMatrix);

    const Shader shadowMapVS("lightTransform.vert", GL_VERTEX_SHADER);
    const Shader shadowMapFS("nothing.frag", GL_FRAGMENT_SHADER);
    ShaderProgram shadowMapSP(shadowMapVS, shadowMapVS);

    shadowMapSP.addUniform(modelUniform);
    shadowMapSP.addUniform(lightSpaceUniform);
    sp.addUniform(lightSpaceUniform);

    // shadow map displaying stuff //////////
    // FBO stuff
    const Shader smfboFS("smfbo.frag", GL_FRAGMENT_SHADER);
    ShaderProgram smfboSP(fboVS, smfboFS);
    smfboSP.use();

    // put the sm texture handle into a SSBO
    const auto smfboTexHandle = shadowTexture.getHandle();
    Buffer smfboTexHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    smfboTexHandleBuffer.setStorage(std::array<GLuint64, 1>{smfboTexHandle}, GL_DYNAMIC_STORAGE_BIT);
    smfboTexHandleBuffer.bindBase(7);
    bool displayShadowMap = false;

    // regular stuff
    Timer timer;

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);

    std::vector<glm::vec3> rotations(5, glm::vec3(0.0f));

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
        sp.showReloadShaderGUI(vs, fs, "Forward Lighting");
        fboSP.showReloadShaderGUI(fboVS, fboFS, "FBO");

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("FBO settings");
            if (ImGui::Checkbox("Render to FBO", &useFBO));
            if (ImGui::Checkbox("Display Shadow Map", &displayShadowMap));
            if (ImGui::Checkbox("Rotate Model", &rotate));
            ImGui::End();
        }

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            //ImGui::SetNextWindowPos(ImVec2(20, 150));
            ImGui::Begin("Lights settings");
            for (int i = 0; i < lvec.size(); ++i)
            {
                std::stringstream n;
                n << i;
                ImGui::Text((std::string("Light ") + n.str()).c_str());
                if (ImGui::SliderFloat3((std::string("Color ") + n.str()).c_str(), value_ptr(lvec.at(i).col), 0.0f, 1.0f))
                {
                    const auto colOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, col);
                    lightBuffer.setContentSubData(lvec.at(i).col, colOffset);
                }
                if (ImGui::SliderFloat((std::string("Cutoff ") + n.str()).c_str(), &lvec.at(i).spot_cutoff, 0.0f, 0.5f))
                {
                    const auto spotCutoffOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_cutoff);
                    lightBuffer.setContentSubData(lvec.at(i).spot_cutoff, spotCutoffOffset);
                }
                if (ImGui::SliderFloat((std::string("Exponent ") + n.str()).c_str(), &lvec.at(i).spot_exponent, 0.0f, 100.0f))
                {
                    const auto spotCutoffExpOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_exponent);
                    lightBuffer.setContentSubData(lvec.at(i).spot_exponent, spotCutoffExpOffset);
                }
                if (ImGui::SliderFloat3((std::string("Rotate ") + n.str()).c_str(), value_ptr(rotations.at(i)), 0.0f, 360.0f))
                {
                    const auto posOffset = i * sizeof(lvec.at(i));
                    const glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(rotations.at(i).x), glm::vec3(1.0f, 0.0f, 0.0f));
                    const glm::mat4 rotxy = glm::rotate(rotx, glm::radians(rotations.at(i).y), glm::vec3(0.0f, 1.0f, 0.0f));
                    const glm::mat4 rotxyz = glm::rotate(rotxy, glm::radians(rotations.at(i).z), glm::vec3(0.0f, 0.0f, 1.0f));
                    const glm::vec3 newPos = rotxyz * lvec.at(i).pos;
                    lightBuffer.setContentSubData(newPos, posOffset);
                    lvec.at(i).spot_direction = normalize(glm::vec3(0.0f) - newPos);
                    const auto spotDirOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_direction);
                    lightBuffer.setContentSubData(lvec.at(i).spot_direction, spotDirOffset);

                    lightView = lookAt(newPos,
                                       glm::vec3(0.0f), // aimed at the center
                                       glm::vec3(0.0f, 1.0f, 0.0f));

                    lightSpaceMatrix = lightProjection * lightView;
                    lightSpaceUniform->setContent(lightSpaceMatrix);
                }
                // maps memory to access it by GUI -- probably very bad performance-wise
                const auto positionOffset = i * sizeof(lvec.at(i));
                //lightBuffer.bind();
                float* ptr = lightBuffer.mapBufferContent<float>(sizeof(float) * 3, positionOffset, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
                ImGui::SliderFloat3((std::string("Position (conflicts rotation) ") + n.str()).c_str(), ptr, -30.0f, 30.0f);
                lightBuffer.unmapBuffer();
            }
            ImGui::End();
        }

        // shadow mapping pass ///////
        {
            // set shadow mapping settings
            shadowMapSP.use();
            glViewport(0, 0, shadowWidth, shadowHeight);
            shadowMapFBO.bind();
            glClear(GL_DEPTH_BUFFER_BIT);
            glCullFace(GL_FRONT);

            // render scene to shadow map
            modelUniform->setContent(bunny->getModelMatrix());
            MaterialIDUniform->setContent(bunny->getMaterialID());
            shadowMapSP.updateUniforms();
            bunny->draw();
            modelUniform->setContent(plane.getModelMatrix());
            MaterialIDUniform->setContent(plane.getMaterialID());
            shadowMapSP.updateUniforms();
            plane.draw();

            // reset settings
            shadowMapFBO.unbind();
            glViewport(0, 0, width, height);
            glCullFace(GL_BACK);
        }
        // end shadow mapping pass /////////////

        //ImGui::ShowTestWindow();
        if (useFBO)
            fbo.bind(); // render into fbo

        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.update(window);

        viewUniform->setContent(camera.getView());
        sp.use();

        // prepare first mesh (bunny)
        modelUniform->setContent(bunny->getModelMatrix());
        MaterialIDUniform->setContent(bunny->getMaterialID());
        sp.updateUniforms();

        bunny->draw();

        // prepare plane
        modelUniform->setContent(plane.getModelMatrix());
        MaterialIDUniform->setContent(plane.getMaterialID());
        sp.updateUniforms();

        plane.draw();

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
