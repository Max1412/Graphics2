#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include <memory>

#include "Utils/UtilCollection.h"
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
    std::vector<glm::vec3> vertices = meshes.at(0).getVertices();
    std::vector<glm::vec3> normals = meshes.at(0).getNormals();
    std::vector<unsigned int> indices = meshes.at(0).getIndices();


    Buffer vBuffer;
    vBuffer.setData(vertices, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    Buffer nBuffer;
    nBuffer.setData(normals, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    Buffer iBuffer;
    iBuffer.setData(indices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

    VertexArray vao;
    vao.connectBuffer(vBuffer, 0, 3, GL_FLOAT, GL_FALSE);
    vao.connectBuffer(nBuffer, 1, 3, GL_FLOAT, GL_FALSE);
    vao.bind();
    iBuffer.bind();

    sp.use();

    SimpleTrackball camera(width, height, 10.0f);
    glm::mat4 view = camera.getView();

    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / (float)height, 1.0f, 1000.0f);
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
    model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));

    auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
    auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(modelUniform);

    LightInfo li;
    li.Position = glm::vec4(0.0f, 0.0f, 20.0f, 0.0f);
    li.Intensity = glm::vec3(0.3f);

    MaterialInfo m;
    m.Ka = glm::vec3(1.0f, 1.0f, 1.0f);
    m.Kd = glm::vec3(0.9f, 0.9f, 0.9f);
    m.Ks = glm::vec3(0.5f, 0.5f, 0.5f);
    m.Shininess = 15.0f;

    Buffer lightBuffer;
    lightBuffer.setData(std::vector<LightInfo>{li}, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    lightBuffer.bindBase(0);

    Buffer materialBuffer;
    materialBuffer.setData(std::vector<MaterialInfo>{m}, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
    materialBuffer.bindBase(1);

    float angle = 0.01f;

    glEnable(GL_DEPTH_TEST);

    glm::vec4 clear_color(0.1f);

    bool flat = false;
    auto flatUniform = std::make_shared<Uniform<bool>>("useFlat", flat);
    sp.addUniform(flatUniform);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("ImGUI test window");
            ImGui::Checkbox("Flat Shading", &flat);
            ImGui::End();
        }

        ImGui::ShowTestWindow();


        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.update(window);

        modelUniform->setContent(camera.getView());
        flatUniform->setContent(flat);

        sp.updateUniforms();

        //glDrawArrays(GL_TRIANGLES, 0, numVertices);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        ImGui::Render();
        glfwSwapBuffers(window);
    }

    vBuffer.del();
    nBuffer.del();
    iBuffer.del();
    vao.del();
    sp.del();
    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
