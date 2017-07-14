#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include <memory>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/VertexArray.h"
#include "Rendering/Uniform.h"
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/SimpleTrackball.h"

#include "imgui/imgui_impl_glfw_gl3.h"

const unsigned int width = 1600;
const unsigned int height = 900;

struct LightInfo {
    glm::vec4 Position; // Light position in eye coords.
    glm::vec3 Intensity; // Ambient light intensity
    float pad;
};

struct MaterialInfo {
    glm::vec3 Ka; // Ambient reflectivity
    float pad4;
    glm::vec3 Kd; // Diffuse reflectivity
    float pad5;
    glm::vec3 Ks; // Specular reflectivity
    float Shininess; // Specular shininess factor
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

    std::array<glm::vec3, 4> planePositions = {
        glm::vec3(-1, 0, -1), glm::vec3(-1, 0, 1), glm::vec3(1, 0, 1), glm::vec3(1, 0, -1)
    };

    std::array<glm::vec3, 4> planeNormals = {
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)
    };

    std::array<unsigned int, 6> planeIndices = {
        0, 1, 2, 
        2, 3, 0
    };

    Buffer PlaneVertexBuffer;
    PlaneVertexBuffer.setData(planePositions, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    Buffer PlaneNormalBuffer;
    PlaneNormalBuffer.setData(planeNormals, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    Buffer PlaneIndexBuffer;
    PlaneIndexBuffer.setData(planeIndices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

    VertexArray PlaneVao;
    PlaneVao.connectBuffer(PlaneVertexBuffer, 0, 3, GL_FLOAT, GL_FALSE);
    PlaneVao.connectBuffer(PlaneNormalBuffer, 1, 3, GL_FLOAT, GL_FALSE);
    PlaneVao.bind();
    PlaneIndexBuffer.bind();

    sp.use();

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

    auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
    auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(modelUniform);
    
    // "generate" lights
    std::vector<LightInfo> lvec;
    for (int i = 0; i < 5; i++) {
        LightInfo li;
        glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), glm::radians(i*(360.0f / 3.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
        li.Position = rotMat * glm::vec4(i*3.0f, i*3.0f, i*3.0f, 0.0f);
        li.Intensity = glm::normalize(glm::vec3((i) % 5,(i+1) % 5, (i + 2) % 5)) / 2.0f;
        if (i % 2) {
            li.Intensity = glm::normalize(glm::vec3((i-1) % 5, (i) % 5, (i + 1) % 5)) / 2.0f;
            li.Intensity = glm::normalize(glm::vec3(1.0f) - li.Intensity) / 2.0f;
        }
        std::cout << glm::to_string(li.Intensity) << std::endl;
        lvec.push_back(li);
    }

    // set up materials
    std::vector<MaterialInfo> mvec;
    MaterialInfo m;
    m.Ka = glm::vec3(0.3f);
    m.Kd = glm::vec3(0.3f);
    m.Ks = glm::vec3(0.9f);
    m.Shininess = 100.0f;
    mvec.push_back(m);

    MaterialInfo m2;
    m2.Ka = glm::vec3(0.3f);
    m2.Kd = glm::vec3(0.3f);
    m2.Ks = glm::vec3(0.3f);
    m2.Shininess = 20.0f;
    mvec.push_back(m2);

    // first material is for bunny
    bunny.setMaterialID(0);

    // create buffers for materials and lights
    Buffer lightBuffer;
    lightBuffer.setData(lvec, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    lightBuffer.bindBase(0);

    Buffer materialBuffer;
    materialBuffer.setData(mvec, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    materialBuffer.bindBase(1);

    glm::vec4 clear_color(0.1f);

    bool flat = false;
    bool lastFlat = flat;
    bool toon = false;
    bool lastToon = toon;
    int levels = 3;
    int lastLevels = levels;

    int materialID = 0;

    auto flatUniform = std::make_shared<Uniform<bool>>("useFlat", flat);
    auto toonUniform = std::make_shared<Uniform<bool>>("useToon", toon);
    auto levelsUniform = std::make_shared<Uniform<int>>("levels", levels);
    auto MaterialIDUniform = std::make_shared<Uniform<int>>("mID", materialID);

    sp.addUniform(flatUniform);
    sp.addUniform(toonUniform);
    sp.addUniform(levelsUniform);
    sp.addUniform(MaterialIDUniform);

    Timer timer;

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);


    // render loop
    while (!glfwWindowShouldClose(window)) {

        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        {
            ImGui::SetNextWindowSize(ImVec2(250, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("Lighting settings");
            ImGui::Checkbox("Flat Shading", &flat);
            if (flat != lastFlat) {
                flatUniform->setContent(flat);
                lastFlat = flat;
            }
            ImGui::Checkbox("Toon Shading", &toon);
            if (toon != lastToon) {
                toonUniform->setContent(toon);
                lastToon = toon;
            }
            if (toon) {
                ImGui::SliderInt("Toon Shading Levels", &levels, 1, 5);
                if (levels != lastLevels) {
                    levelsUniform->setContent(levels);
                    lastLevels = levels;
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
        modelUniform->setContent(PlaneModel);
        MaterialIDUniform->setContent(1);
        sp.updateUniforms();

        PlaneVao.bind();
        PlaneIndexBuffer.bind();
        glDrawElements(GL_TRIANGLES, planeIndices.size(), GL_UNSIGNED_INT, 0);

        timer.stop();
        timer.drawGuiWindow();

        ImGui::Render();
        glfwSwapBuffers(window);
    }

    sp.del();
    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
