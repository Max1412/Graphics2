#include <GL/glew.h>

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

#include "imgui/imgui_impl_glfw_gl3.h"

const unsigned int width = 1600;
const unsigned int height = 900;

struct LightInfo {
    glm::vec4 pos; //pos.w=0 dir., pos.w=1 point light
    glm::vec3 col;
    float spot_cutoff; //no spotlight if cutoff=0
    glm::vec3 spot_direction;
    float spot_exponent;
};

struct MaterialInfo {
    glm::vec3 diffColor;
    float kd;
    glm::vec3 specColor;
    float ks;
    float shininess;
    float kt;
    float pad1, pad2;
};

struct FogInfo {
    glm::vec3 col;
    float start;
    float end;
    float density;
    int mode;
    float pad;
};

int main() {
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Demo 2");
    glfwSwapInterval(0);
    // init glew and check for errors
    util::initGLEW();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
    ImGui_ImplGlfwGL3_Init(window, true);

    // get list of OpenGL extensions (can be searched later if needed)
    std::vector<std::string> extensions = util::getGLExtenstions();

    Shader vs("quadSDF.vert", GL_VERTEX_SHADER);
    Shader fs("traceSDF.frag", GL_FRAGMENT_SHADER);
    ShaderProgram sp(vs, fs);


    std::vector<glm::vec2> quadData = {
        { -1.0, -1.0 },
        { 1.0, -1.0 },
        { -1.0, 1.0 },
        { -1.0, 1.0 },
        { 1.0, -1.0 },
        { 1.0, 1.0 }
    };

    Buffer QuadBuffer(GL_ARRAY_BUFFER);
    QuadBuffer.setStorage(quadData, GL_DYNAMIC_STORAGE_BIT);
    VertexArray quadVAO;
    quadVAO.connectBuffer(QuadBuffer, 0, 2, GL_FLOAT, GL_FALSE);
    quadVAO.bind();

    sp.use();

    // create matrices for uniforms
    SimpleTrackball camera(width, height, 10.0f);
    glm::mat4 view = camera.getView();

    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / (float)height, 1.0f, 1000.0f);
    
    // create matrix uniforms and add them to the shader program
    auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);

    glm::vec3 ambient(0.5f);
    auto ambientLightUniform = std::make_shared<Uniform<glm::vec3>>("lightAmbient", ambient);
    sp.addUniform(ambientLightUniform);

    // "generate" lights
    std::vector<LightInfo> lvec;
    for (int i = 0; i < 2; i++) {
        LightInfo li;
        glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), glm::radians(i*(360.0f / 5.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
        li.pos = rotMat * (glm::vec4((i + 1)*3.0f, (i + 1)*3.0f, (i + 1)*3.0f, 1.0f) + glm::vec4(0.0001f, 0.0001f, 0.0001f, 0.0f));
        li.col = glm::normalize(glm::vec3((i) % 5, (i + 1) % 5, (i + 2) % 5));
        if (i % 2) {
            li.col = glm::normalize(glm::vec3((i - 1) % 5, (i) % 5, (i + 1) % 5));
            li.col = glm::normalize(glm::vec3(1.0f) - li.col);
        }
        std::cout << glm::to_string(li.col) << std::endl;
        if (i == 3) {
            li.spot_cutoff = 0.1f;
        }
        else {
            li.spot_cutoff = 0.0f;
        }
        li.spot_direction = glm::normalize(glm::vec3(0.0f) - glm::vec3(li.pos));
        li.spot_exponent = 1.0f;


        lvec.push_back(li);
    }
    // create buffers for materials and lights
    Buffer lightBuffer(GL_SHADER_STORAGE_BUFFER);
    lightBuffer.setStorage(lvec, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
    lightBuffer.bindBase(0);

    glm::vec4 clear_color(0.1f);
    std::vector<glm::vec3> rotations(5, glm::vec3(0.0f));

    Timer timer;

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);

    // render loop
    while (!glfwWindowShouldClose(window)) {

        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        sp.showReloadShaderGUI(vs, fs);

        //ImGui::ShowTestWindow();

        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            //ImGui::SetNextWindowPos(ImVec2(20, 150));
            ImGui::Begin("Lights settings");
            for (int i = 0; i < lvec.size(); ++i) {
                std::stringstream n;
                n << i;
                ImGui::Text((std::string("Light ") + n.str()).c_str());
                if (ImGui::SliderFloat3((std::string("Color ") + n.str()).c_str(), glm::value_ptr(lvec.at(i).col), 0.0f, 1.0f)) {
                    auto colOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, col);
                    lightBuffer.setContentSubData(lvec.at(i).col, colOffset);
                }
                if (ImGui::SliderFloat((std::string("Cutoff ") + n.str()).c_str(), &lvec.at(i).spot_cutoff, 0.0f, 0.5f)) {
                    auto spotCutoffOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_cutoff);
                    lightBuffer.setContentSubData(lvec.at(i).spot_cutoff, spotCutoffOffset);
                }
                if (ImGui::SliderFloat((std::string("Exponent ") + n.str()).c_str(), &lvec.at(i).spot_exponent, 0.0f, 100.0f)) {
                    auto spotCutoffExpOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_exponent);
                    lightBuffer.setContentSubData(lvec.at(i).spot_exponent, spotCutoffExpOffset);
                }
                if (ImGui::SliderFloat3((std::string("Rotate ") + n.str()).c_str(), glm::value_ptr(rotations.at(i)), 0.0f, 360.0f)) {
                    auto posOffset = i * sizeof(lvec.at(i));
                    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(rotations.at(i).x), glm::vec3(1.0f, 0.0f, 0.0f));
                    glm::mat4 rotxy = glm::rotate(rotx, glm::radians(rotations.at(i).y), glm::vec3(0.0f, 1.0f, 0.0f));
                    glm::mat4 rotxyz = glm::rotate(rotxy, glm::radians(rotations.at(i).z), glm::vec3(0.0f, 0.0f, 1.0f));
                    glm::vec3 newPos = rotxyz * lvec.at(i).pos;
                    lightBuffer.setContentSubData(newPos, posOffset);
                    lvec.at(i).spot_direction = glm::normalize(glm::vec3(0.0f) - newPos);
                    auto spotDirOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_direction);
                    lightBuffer.setContentSubData(lvec.at(i).spot_direction, spotDirOffset);
                }
                // maps memory to access it by GUI -- probably very bad performance-wise
                auto positionOffset = i * sizeof(lvec.at(i));
                //lightBuffer.bind();
                float *ptr = lightBuffer.mapBufferContent<float>(sizeof(float) * 3, positionOffset, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
                ImGui::SliderFloat3((std::string("Position (conflicts rotation) ") + n.str()).c_str(), ptr, -30.0f, 30.0f);
                lightBuffer.unmapBuffer();
            }
            ImGui::End();
        }

        camera.update(window);
        viewUniform->setContent(camera.getView());
        sp.updateUniforms();

        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(quadData.size()));

        timer.stop();
        timer.drawGuiWindow(window);

        ImGui::Render();
        glfwSwapBuffers(window);
    }

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
