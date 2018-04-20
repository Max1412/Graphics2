#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
#include "Rendering/IBLCubemapMaker.h"

const unsigned int width = 1600;
const unsigned int height = 900;
const int numSpheres = 3;

struct PBRMaterial
{
    glm::vec4 baseColor;
    glm::vec3 F0;
    float metalness;
    float roughness;
    int32_t pad, pad1, pad2;
};

struct Light
{
    glm::vec4 position;
    glm::vec4 color;
};

int main() 
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(width, height, "Rapid Testing Executable");
    glfwSwapInterval(0);
    // init glew and check for errors
    util::initGLEW();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
	ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    // get list of OpenGL extensions (can be searched later if needed)
    std::vector<std::string> extensions = util::getGLExtenstions();
    
    Shader vs("pbr1.vert", GL_VERTEX_SHADER);
    Shader fs("pbr3.frag", GL_FRAGMENT_SHADER);
    ShaderProgram sp(vs, fs);

    ModelImporter mi("sphere.obj");
    auto meshes = mi.getMeshes();
    auto sphere = meshes.at(0);
    
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 1000.0f);
    glm::mat4 model(1.0f);
    //model = glm::translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
    sphere->setModelMatrix(model);
    sphere->setMaterialID(0);

    // create matrices for uniforms
    SimpleTrackball camera(width, height, 10.0f);
    glm::mat4 view = camera.getView();
    glm::vec3 camPos = camera.getPosition();

    auto camPosUniform = std::make_shared<Uniform<glm::vec3>>("camPos", camPos);
    sp.addUniform(camPosUniform);

    // create matrix uniforms and add them to the shader program
    const auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
    auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
    auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);

    sp.addUniform(projUniform);
    sp.addUniform(viewUniform);
    sp.addUniform(modelUniform);

    // shading uniforms
    auto MaterialIDUniform = std::make_shared<Uniform<int>>("matIndex", sphere->getMaterialID());

    sp.addUniform(MaterialIDUniform);

    std::vector<PBRMaterial> mvec;// (numSpheres * numSpheres, PBRMaterial{ glm::vec4(1.0f), 0.04f, 0.5f, 0.5f, 0 });
    for(int i = 0; i < numSpheres; i++)
    {
        for (int j = 0; j < numSpheres; j++)
        {
            const float roughness = std::min(0.025f + j / static_cast<float>(numSpheres), 1.0f);
            const float metalness = i / static_cast<float>(numSpheres);
            mvec.push_back(PBRMaterial{ glm::vec4(0.5f, 0.0f, 0.0f, 1.0f), glm::vec3(0.04f), metalness, roughness, 0 });
        }

    }

    Buffer materialBuffer(GL_SHADER_STORAGE_BUFFER);
    materialBuffer.setStorage(mvec, GL_DYNAMIC_STORAGE_BIT);
    materialBuffer.bindBase(1);

    std::vector<Light> lightVec = 
	{
        Light{ glm::vec4(-10.0f, 10.0f, 10.0f, 1.0f), glm::vec4(300.0f) },
        Light{ glm::vec4(10.0f, 10.0f, 10.0f, 1.0f), glm::vec4(300.0f) },
        Light{ glm::vec4(-10.0f, -10.0f, 10.0f, 1.0f), glm::vec4(300.0f) },
        Light{ glm::vec4(10.0f, -10.0f, 10.0f, 1.0f), glm::vec4(300.0f) }
    };
    Buffer lightBuffer(GL_SHADER_STORAGE_BUFFER);
    lightBuffer.setStorage(lightVec, GL_DYNAMIC_STORAGE_BIT);
    lightBuffer.bindBase(2);

    // IBL STUFF
    IBLCubemapMaker cubemapMaker(RESOURCES_PATH + std::string("/Newport_Loft/Newport_Loft_Ref.hdr"));
    glViewport(0, 0, width, height);

    Buffer lightingTextureBuffer(GL_SHADER_STORAGE_BUFFER);
    lightingTextureBuffer.setStorage(std::array<GLuint64, 3>{cubemapMaker.getIrradianceCubemap().getHandle(), cubemapMaker.getSpecularCubemap().getHandle(), cubemapMaker.getBRDFLUT().getHandle()}, GL_DYNAMIC_STORAGE_BIT);
    lightingTextureBuffer.bindBase(7);

    SkyBoxCube cube;

    bool useIBLirradiance = true;
    auto useIBLirradianceUniform = std::make_shared<Uniform<bool>>("useIBLirradiance", useIBLirradiance);
    sp.addUniform(useIBLirradianceUniform);

    // regular stuff
    Timer timer;

    const glm::vec4 clearColor(0.2f, 0.2f, 0.2f, 1.0f);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);

    // render loop
    while (!glfwWindowShouldClose(window))
    {

        timer.start();

        glfwPollEvents();
        ImGui_ImplGlfwGL3_NewFrame();
        sp.showReloadShaderGUI(vs, fs, "PBR");
        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            //ImGui::SetNextWindowPos(ImVec2(20, 150));
            ImGui::Begin("Material settings");
            for (int i = 0; i < mvec.size(); ++i) 
			{
                std::stringstream n;
                n << i;
                ImGui::Text((std::string("Material ") + n.str()).c_str());
                if (ImGui::SliderFloat4((std::string("Color ") + n.str()).c_str(), glm::value_ptr(mvec.at(i).baseColor), 0.0f, 1.0f)) 
				{
                    const auto colOffset = i * sizeof(mvec.at(i)) + offsetof(PBRMaterial, baseColor);
                    materialBuffer.setContentSubData(mvec.at(i).baseColor, colOffset);
                }
                if (ImGui::SliderFloat3((std::string("F0 ") + n.str()).c_str(), glm::value_ptr(mvec.at(i).F0), 0.0f, 0.1f)) 
				{
                    const auto F0offset = i * sizeof(mvec.at(i)) + offsetof(PBRMaterial, F0);
                    materialBuffer.setContentSubData(mvec.at(i).F0, F0offset);
                }
                if (ImGui::SliderFloat((std::string("Metalness ") + n.str()).c_str(), &mvec.at(i).metalness, 0.0f, 1.0f)) 
				{
                    const auto metalnessOffset = i * sizeof(mvec.at(i)) + offsetof(PBRMaterial, metalness);
                    materialBuffer.setContentSubData(mvec.at(i).metalness, metalnessOffset);
                }
                if (ImGui::SliderFloat((std::string("Roughness ") + n.str()).c_str(), &mvec.at(i).roughness, 0.0f, 1.0f)) 
				{
                    const auto roughnessOffset = i * sizeof(mvec.at(i)) + offsetof(PBRMaterial, roughness);
                    materialBuffer.setContentSubData(mvec.at(i).roughness, roughnessOffset);
                }
            }

            ImGui::End();
        }

        {
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("Rendering Settings");
            if (ImGui::Checkbox("Use IBL irradiance", &useIBLirradiance))
                useIBLirradianceUniform->setContent(useIBLirradiance);
            ImGui::End();
        }

        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.update(window);
        camPosUniform->setContent(camera.getPosition());

        viewUniform->setContent(camera.getView());
        sp.use();

        // prepare first mesh (bunny)
        //modelUniform->setContent(sphere->getModelMatrix());
        //MaterialIDUniform->setContent(sphere->getMaterialID());
        //sp.updateUniforms();

        //sphere->draw();

        // draw 5x5 grid of spheres
        float spacing = 2.5;
        for(int i = -numSpheres / 2; i < numSpheres / 2 + 1; i++)
        {
            for(int j = -numSpheres / 2; j < numSpheres / 2 + 1; j++)
            {
                modelUniform->setContent(glm::translate(sphere->getModelMatrix(), glm::vec3(spacing * i, spacing * j, 0.0f)));
                MaterialIDUniform->setContent(numSpheres * (j + numSpheres / 2) + (i + numSpheres / 2));
                sp.updateUniforms();
                sphere->draw();          
            }
        }

        {
            cubemapMaker.draw(camera.getView(), proj);
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
