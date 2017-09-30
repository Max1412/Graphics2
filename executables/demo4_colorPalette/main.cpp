#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/Uniform.h"
#include "Rendering/Mesh.h"

#define STB_IMAGE_IMPLEMENTATION
#include "IO/stb_image.h"

#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/Texture.h"
#include "Rendering/Image.h"

const unsigned int width = 1600;
const unsigned int height = 900;

struct Centroid
{
    GLuint64 imageHandle;
    glm::vec3 color;
    int pad1, pad2, pad3;
};

int main() {
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Color Palette");
    glfwSwapInterval(0);
    // init glew and check for errors
    util::initGLEW();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
    ImGui_ImplGlfwGL3_Init(window, true);

    // get list of OpenGL extensions (can be searched later if needed)
    auto extensions = util::getGLExtenstions();

    Shader vs("texSFQ.vert", GL_VERTEX_SHADER);
    Shader fs("texSFQ.frag", GL_FRAGMENT_SHADER);
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

    const std::vector<glm::vec2> quadTexCoordData = {
        { 0.0, 0.0 },
        { 1.0, 0.0 },
        { 0.0, 1.0 },
        { 0.0, 1.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 }
    };

    Buffer QuadBuffer(GL_ARRAY_BUFFER);
    QuadBuffer.setData(quadData, GL_STATIC_DRAW);

    Buffer TexCoordBuffer(GL_ARRAY_BUFFER);
    TexCoordBuffer.setData(quadTexCoordData, GL_STATIC_DRAW);

    VertexArray quadVAO;
    quadVAO.connectBuffer(QuadBuffer, 0, 2, GL_FLOAT, GL_FALSE);
    quadVAO.connectBuffer(TexCoordBuffer, 1, 2, GL_FLOAT, GL_FALSE);

    quadVAO.bind();

    Timer timer;

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);

    Texture tex;
    tex.loadFromFile(RESOURCES_PATH + std::string("/image.png"));
    tex.generateHandle();
   
    const auto textureHandle = tex.getHandle();

    // put the texture handle into a SSBO
    Buffer textureHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    textureHandleBuffer.setData(std::array<GLuint64, 1>{textureHandle}, GL_STATIC_DRAW);
    textureHandleBuffer.bindBase(0);

    // create a DSA-image (immutable) and set its content. Size: Window size
    Image image;
    image.initWithoutData(width, height, GL_R8);
    const unsigned char clearVal = 0;
    image.clearTexture(GL_RED, GL_UNSIGNED_BYTE, clearVal);
    
    image.generateImageHandle(GL_R8UI);
    const auto imageHandle = image.getHandle();
   

    // put the image handle into a SSBO
    Buffer imageHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    imageHandleBuffer.setData(std::array<GLuint64, 1>{imageHandle}, GL_STATIC_DRAW);
    imageHandleBuffer.bindBase(1);


    // render loop
    while (!glfwWindowShouldClose(window)) {

        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();

        sp.showReloadShaderGUI(vs, fs);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        sp.updateUniforms();

        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(quadData.size()));

        // leave this in for now
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // debugging: looking at the image content
        //std::vector<unsigned char> imageContent;
        //imageContent.resize(sizeof(unsigned char) * width * height);
        //glGetTextureImage(image, 0, GL_RED, GL_UNSIGNED_BYTE, imageContent.size() * sizeof(unsigned char), imageContent.data());

        //for(auto& pixel : imageContent)
        //{
        //    if(pixel != 1)
        //    {
        //        std::cout << "Value other than 1: " << pixel << '\n';
        //    }
        //}

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
