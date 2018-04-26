#include <glbinding/gl/gl.h>
#include "Rendering/Quad.h"
#include "Rendering/SimpleTrackball.h"
#include <glm/gtc/matrix_transform.inl>
#include "Rendering/SparseVoxelOctree.h"
#include "IO/ModelImporter.h"
using namespace gl;

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Uniform.h"
#include "Rendering/Mesh.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include <iostream>

const unsigned int width = 800;
const unsigned int height = 800;

int main()
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Demo 6 - Octree");
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

    glViewportIndexedf(0, 0.f, 0.f, static_cast<float>(width), static_cast<float>(height));

    ModelImporter mi("bunny.obj");
    auto meshes = mi.getMeshes();
    auto bunny = meshes.at(0);
    std::vector<std::shared_ptr<Mesh>> scene;
    scene.push_back(bunny);

    SparseVoxelOctree svo(scene, 5);

    SimpleTrackball cam(width, height, 5);

    Shader vs("SparseVoxelOctree/sfq_rayDir.vert", GL_VERTEX_SHADER);
    Shader fs("SparseVoxelOctree/TraceSparseVoxelOctree.frag", GL_FRAGMENT_SHADER);
    ShaderProgram sp(vs, fs);
    sp.use();

    Quad q;

    auto u_view = std::make_shared<Uniform<glm::mat4>>("viewMatrix", cam.getView());
    auto u_projection = std::make_shared<Uniform<glm::mat4>>("projectionMatrix", glm::perspective(glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 1000.0f));
    auto u_camPos = std::make_shared<Uniform<glm::vec3>>("camPosition", cam.getPosition());
    auto u_bmin = std::make_shared<Uniform<glm::vec3>>("bmin", svo.getBMin());
    auto u_bmax = std::make_shared<Uniform<glm::vec3>>("bmax", svo.getBMax());
    auto u_maxLevel = std::make_shared<Uniform<int>>("maxLevel", 10);
    sp.addUniform(u_view);
    sp.addUniform(u_projection);
    sp.addUniform(u_camPos);
    sp.addUniform(u_bmin);
    sp.addUniform(u_bmax);
    sp.addUniform(u_maxLevel);

    glm::vec4 clear_color(0.1f);

    Timer timer;

    int maxLevelRender = 10; double keyTimeout = 0; //only for octree level debugging!

    // render loop
    while (!glfwWindowShouldClose(window))
    {
        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        sp.showReloadShaderGUI({ vs, fs });

        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //only for level debugging!
        if ((glfwGetTime() - keyTimeout) * 1000.0 > 100.0) {
            if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS) {
                maxLevelRender--;
                std::cout << "Max Octree traversion depth: " << maxLevelRender << '\n';
                keyTimeout = glfwGetTime();
            }
            if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS) {
                maxLevelRender++;
                std::cout << "Max Octree traversion depth: " << maxLevelRender << '\n';
                keyTimeout = glfwGetTime();
            }
            keyTimeout = glfwGetTime();
        }
        u_maxLevel->setContent(maxLevelRender);

        std::cout << "SVO update time: " << util::timeCall([&]() {svo.update();}) << "ms \n";

        cam.update(window);
        u_view->setContent(cam.getView());
        u_camPos->setContent(cam.getPosition());
        sp.use();
        svo.bind();

        q.draw();

        timer.stop();
        timer.drawGuiWindow(window);

        ImGui::Render();
        ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    std::cout << std::endl;

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
