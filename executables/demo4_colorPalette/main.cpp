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

const unsigned int width = 1600;
const unsigned int height = 900;

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

    // load an image from disk
    auto imageFilename(RESOURCES_PATH + std::string("/image.png"));
    int imageWidth, imageHeight, numChannels;
    const auto imageData = stbi_load(imageFilename.c_str(), &imageWidth, &imageHeight, &numChannels, 4);

    if (!imageData)
        throw std::runtime_error("Image couldn't be loaded");
    
    // create a DSA-texture (immutable) and set its content. Size: actual texture size
    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(texture, 1, GL_RGBA8, imageWidth, imageHeight);
    glTextureSubImage2D(texture, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, imageData);
    const auto textureHandle = glGetTextureHandleARB(texture);
    if (textureHandle == 0)
        throw std::runtime_error("Texture handle could not be returned");
    glMakeTextureHandleResidentARB(textureHandle);

    // put the texture handle into a SSBO
    Buffer textureHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    textureHandleBuffer.setData(std::array<GLuint64, 1>{textureHandle}, GL_STATIC_DRAW);
    textureHandleBuffer.bindBase(0);

    // create a DSA-image (immutable) and set its content. Size: Window size
    GLuint image;
    glCreateTextures(GL_TEXTURE_2D, 1, &image);
    glTextureParameteri(image, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(image, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureStorage2D(image, 1, GL_R8, width, height);

    // set all image contents to 0
    unsigned char clearVal = 0;
    glClearTexImage(image, 0, GL_RED, GL_UNSIGNED_BYTE, &clearVal);

    // do bindless image stuff
    const auto imageHandle = glGetImageHandleARB(image, 0, GL_FALSE, 0, GL_R8UI);
    if (imageHandle == 0)
        throw std::runtime_error("Texture handle could not be returned");
    glMakeImageHandleResidentARB(imageHandle, GL_READ_WRITE);

    // put the image handle into a SSBO
    Buffer imageHandleBuffer(GL_SHADER_STORAGE_BUFFER);
    imageHandleBuffer.setData(std::array<GLuint64, 1>{imageHandle}, GL_STATIC_DRAW);
    imageHandleBuffer.bindBase(1);


    // let the cpu data of the image go
    stbi_image_free(imageData);
    

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
