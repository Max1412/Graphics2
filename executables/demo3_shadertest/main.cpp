#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/Uniform.h"
#include "Rendering/Mesh.h"

#include "imgui/imgui_impl_glfw_gl3.h"
#include <iostream>

const unsigned int width = 800;
const unsigned int height = 800;

int main(int argc, char* argv[]) {
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Demo 3");
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

    Shader vs("sfq.vert", GL_VERTEX_SHADER);
    Shader fs("mondrian.frag", GL_FRAGMENT_SHADER);
    ShaderProgram sp(vs, fs);
    sp.use();

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

    auto resolutionUniform = std::make_shared<Uniform<glm::vec2>>("u_resolution", glm::vec2(width, height));
    sp.addUniform(resolutionUniform);

    auto time = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::high_resolution_clock::now();
    auto dur = (time2 - time).count();
    auto duration = dur / 1000000000.0f;
    auto timeUniform = std::make_shared<Uniform<float>>("u_time", duration);
    // catch optimized-out uniforms -- this is for testing purposes only
    try
    {
        sp.addUniform(timeUniform);
    }
    catch (std::runtime_error& err)
    {
        std::cout << "Unused uniform" << '\n';
        std::cout << err.what() << '\n';
    }

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

        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        time2 = std::chrono::high_resolution_clock::now();
        dur = (time2 - time).count();
        duration = dur / 100000000.0f;
        timeUniform->setContent(duration);
        sp.updateUniforms();

        glDrawArrays(GL_TRIANGLES, 0, quadData.size());

        timer.stop();
        timer.drawGuiWindow(window);

        ImGui::Render();
        glfwSwapBuffers(window);
    }
    std::cout << std::endl;

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}