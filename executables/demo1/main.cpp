#define GLEW_STATIC
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

int main(int argc, char* argv[]) {
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Demo 1");
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

    Shader vs("demo1.vert", GL_VERTEX_SHADER);
    Shader fs("demo1.frag", GL_FRAGMENT_SHADER);
    ShaderProgram sp(vs, fs);

    ModelImporter mi("bunny.obj");
    std::vector<Mesh> meshes = mi.getMeshes();
    Mesh bunny = meshes.at(0);

    // create a plane
    std::vector<glm::vec3> planePositions = {
        glm::vec3(-1, 0, -1), glm::vec3(-1, 0, 1), glm::vec3(1, 0, 1), glm::vec3(1, 0, -1)
    };

    std::vector<glm::vec3> planeNormals = {
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)
    };

    std::vector<unsigned> planeIndices = {
        0, 1, 2, 
        2, 3, 0
    };

    Mesh plane(planePositions, planeNormals, planeIndices);

    sp.use();

    // create matrices for uniforms
    SimpleTrackball camera(width, height, 10.0f);
    glm::mat4 view = camera.getView();

    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / (float)height, 1.0f, 1000.0f);
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
    //model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));
    bunny.setModelMatrix(model);

    glm::mat4 PlaneModel(1.0f);
    PlaneModel = glm::translate(PlaneModel, glm::vec3(0.0f, -1.34f, 0.0f));
    PlaneModel = glm::scale(PlaneModel, glm::vec3(5.0f));
    plane.setModelMatrix(PlaneModel);

    // create matrix uniforms and add them to the shader program
    auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
    auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(modelUniform);

    glm::vec3 ambient(0.5f);
    auto ambientLightUniform = std::make_shared<Uniform<glm::vec3>>("lightAmbient", ambient);
    sp.addUniform(ambientLightUniform);

    // "generate" lights
    std::vector<LightInfo> lvec;
    for (int i = 0; i < 5; i++) {
        LightInfo li;
        glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), glm::radians(i*(360.0f / 5.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
        li.pos = rotMat * (glm::vec4(i*3.0f, i*3.0f, i*3.0f, 1.0f) + glm::vec4(0.0001f, 0.0001f, 0.0001f, 0.0f));
        li.col = glm::normalize(glm::vec3((i) % 5,(i+1) % 5, (i + 2) % 5));
        if (i % 2) {
            li.col = glm::normalize(glm::vec3((i-1) % 5, (i) % 5, (i + 1) % 5));
            li.col = glm::normalize(glm::vec3(1.0f) - li.col);
        }
        std::cout << glm::to_string(li.col) << std::endl;
        if(i == 3){
            li.spot_cutoff = 0.1f;
        }
        else {
            li.spot_cutoff = 0.0f;
        }
        li.spot_direction = glm::normalize(glm::vec3(0.0f) - glm::vec3(li.pos));
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
    mvec.push_back(m);

    MaterialInfo m2;
    m2.diffColor = glm::vec3(0.9f);
    m2.kd = 0.3f;
    m2.specColor = glm::vec3(0.9f);
    m2.ks = 0.3f;
    m2.shininess = 20.0f;
    m.kt = 0.0f;
    mvec.push_back(m2);

    // first material is for bunny, second for plane
    bunny.setMaterialID(0);
    plane.setMaterialID(1);

    // create buffers for materials and lights
    Buffer lightBuffer;
    lightBuffer.setData(lvec, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    lightBuffer.bindBase(0);

    Buffer materialBuffer;
    materialBuffer.setData(mvec, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    materialBuffer.bindBase(1);

    FogInfo f;
    f.start = 0.0f;
    f.end = 10.0f;
    f.density = 0.1f;
    f.col = glm::vec3(0.1f);
    f.mode = 3;
    std::vector<FogInfo> fogvec{ f };
    Buffer fogBuffer;
    fogBuffer.setData(fogvec, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    fogBuffer.bindBase(2);

    glm::vec4 clear_color(0.1f);

    // values for GUI-controllable uniforms
    bool flat = false;
    bool toon = false;
    int levels = 3;

    // shading uniforms
    auto flatUniform = std::make_shared<Uniform<bool>>("useFlat", flat);
    auto toonUniform = std::make_shared<Uniform<bool>>("useToon", toon);
    auto levelsUniform = std::make_shared<Uniform<int>>("levels", levels);
    auto MaterialIDUniform = std::make_shared<Uniform<int>>("matIndex", bunny.getMaterialID());

    sp.addUniform(flatUniform);
    sp.addUniform(toonUniform);
    sp.addUniform(levelsUniform);
    sp.addUniform(MaterialIDUniform);

    Timer timer;

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);

    std::vector<glm::vec3> rotations(5, glm::vec3(0.0f));

    // render loop
    while (!glfwWindowShouldClose(window)) {

        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();
        sp.showReloadShaderGUI(vs, fs);
        {
            ImGui::SetNextWindowSize(ImVec2(100, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("Lighting settings");
            if (ImGui::Checkbox("Flat Shading", &flat)) flatUniform->setContent(flat);
            if (ImGui::Checkbox("Toon Shading", &toon)) toonUniform->setContent(toon);
            if (toon) {
                if (ImGui::SliderInt("Toon Shading Levels", &levels, 1, 5))
                    levelsUniform->setContent(levels);
            }
            if (ImGui::SliderInt("Fog Mode", &fogvec.at(0).mode, 0, 3)) {
                size_t fogModeOffset = sizeof(f.col) + sizeof(f.start) + sizeof(f.end) + sizeof(f.density);
                fogBuffer.setPartialContentMapped(fogvec.at(0).mode, fogModeOffset);
            }
            if(ImGui::SliderFloat3("Fog Color", glm::value_ptr(fogvec.at(0).col), 0.0f, 1.0f)){
                fogBuffer.setPartialContentMapped(fogvec.at(0).col, 0);
            }
            if (ImGui::Button("Reset Fog Color")) {
                fogvec.at(0).col = glm::vec3(0.1f);
                fogBuffer.setPartialContentMapped(fogvec.at(0).col, 0);
            }
            ImGui::End();

            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            //ImGui::SetNextWindowPos(ImVec2(20, 150));
            ImGui::Begin("Lights settings");
            for(int i = 0; i < lvec.size(); ++i) {
                std::stringstream n;
                n << i;
                ImGui::Text((std::string("Light ") + n.str()).c_str());
                if (ImGui::SliderFloat3((std::string("Color ") + n.str()).c_str(), glm::value_ptr(lvec.at(i).col), 0.0f, 1.0f)) {
                    size_t colOffset = i * sizeof(lvec.at(i)) + sizeof(lvec.at(i).pos);
                    lightBuffer.setContentSubData(lvec.at(i).col, colOffset);
                }
                if(ImGui::SliderFloat((std::string("Cutoff ") + n.str()).c_str(), &lvec.at(i).spot_cutoff, 0.0f, 0.5f)){
                    size_t spotCutoffOffset = i * sizeof(lvec.at(i)) + sizeof(lvec.at(i).pos) + sizeof(lvec.at(i).col);
                    lightBuffer.setContentSubData(lvec.at(i).spot_cutoff, spotCutoffOffset);
                }
                if (ImGui::SliderFloat((std::string("Exponent ") + n.str()).c_str(), &lvec.at(i).spot_exponent, 0.0f, 100.0f)) {
                    size_t spotCutoffExpOffset = i * sizeof(lvec.at(i)) + sizeof(lvec.at(i).pos) + sizeof(lvec.at(i).col) + sizeof(lvec.at(i).spot_cutoff) + sizeof(lvec.at(i).spot_direction);
                    lightBuffer.setContentSubData(lvec.at(i).spot_exponent, spotCutoffExpOffset);
                }
                if (ImGui::SliderFloat3((std::string("Rotate ") + n.str()).c_str(), glm::value_ptr(rotations.at(i)), 0.0f, 360.0f)) {
                    size_t posOffset = i * sizeof(lvec.at(i));
                    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(rotations.at(i).x), glm::vec3(1.0f, 0.0f, 0.0f));
                    glm::mat4 rotxy = glm::rotate(rotx, glm::radians(rotations.at(i).y), glm::vec3(0.0f, 1.0f, 0.0f));
                    glm::mat4 rotxyz = glm::rotate(rotxy, glm::radians(rotations.at(i).z), glm::vec3(0.0f, 0.0f, 1.0f));
                    glm::vec3 newPos = rotxyz * lvec.at(i).pos;
                    lightBuffer.setContentSubData(newPos, posOffset);
                    lvec.at(i).spot_direction = glm::normalize(glm::vec3(0.0f) - newPos);
                    size_t spotDirOffset = i * sizeof(lvec.at(i)) + sizeof(lvec.at(i).pos) + sizeof(lvec.at(i).col) + sizeof(lvec.at(i).spot_cutoff);
                    lightBuffer.setContentSubData(lvec.at(i).spot_direction, spotDirOffset);
                }

            }
            ImGui::End();
        }

        //ImGui::ShowTestWindow();

        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.update(window);
        viewUniform->setContent(camera.getView());

        // prepare first mesh (bunny)
        modelUniform->setContent(bunny.getModelMatrix());
        MaterialIDUniform->setContent(bunny.getMaterialID());
        sp.updateUniforms();

        bunny.draw();

        // prepare plane
        modelUniform->setContent(plane.getModelMatrix());
        MaterialIDUniform->setContent(plane.getMaterialID());
        sp.updateUniforms();

        plane.draw();

        timer.stop();
        timer.drawGuiWindow(window);

        ImGui::Render();
        glfwSwapBuffers(window);
    }

    lightBuffer.del();
    materialBuffer.del();
    sp.del();
    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
