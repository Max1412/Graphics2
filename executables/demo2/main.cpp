#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
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

    Buffer QuadBuffer;
    QuadBuffer.setData(quadData, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    VertexArray quadVAO;
    quadVAO.connectBuffer(QuadBuffer, 0, 2, GL_FLOAT, GL_FALSE);

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

    glm::vec4 clear_color(0.1f);

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

        camera.update(window);
        viewUniform->setContent(camera.getView());
        sp.updateUniforms();

        glDrawArrays(GL_TRIANGLES, 0, quadData.size());

        timer.stop();
        timer.drawGuiWindow(window);

        ImGui::Render();
        glfwSwapBuffers(window);
    }

    sp.del();
    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
