#include <glbinding/gl/gl.h>
#include "Rendering/Light.h"
#include "Rendering/SimplexNoise.h"
using namespace gl;

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Utils/UtilCollection.h"
#include "Utils/Timer.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/Uniform.h"
#include "Rendering/Image.h"
#include "IO/ModelImporter.h"
#include "Rendering/Mesh.h"
#include "Rendering/VoxelDebugRenderer.h"
#include "Rendering/Pilotview.h"
#include "Rendering/LightManager.h"


#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/FrameBuffer.h"

constexpr int screenWidth = 1600;
constexpr int screenHeight = 900;
constexpr float screenNear = 0.1f;
constexpr float screenFar = 1000.f;
constexpr int gridWidth = screenWidth / 10;
constexpr int gridHeight = screenHeight / 10;
constexpr int gridDepth = 100;
constexpr int groupSize = 4;

constexpr bool renderimgui = true;

struct PlayerCameraInfo
{
    glm::mat4 playerViewMatrix;
    glm::mat4 playerProjMatrix;
    glm::vec3 camPos; float pad;
};

struct FogInfo
{
    glm::vec3 fogAlbedo;
    float fogAnisotropy;
    float fogScatteringCoeff;
    float fogAbsorptionCoeff;
    float pad1, pad2;
};

int main()
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(screenWidth, screenHeight, "Rapid Testing Executable");
    glfwSwapInterval(0);
    // init opengl
    util::initGL();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    Image voxelGrid(GL_TEXTURE_3D, GL_NEAREST, GL_NEAREST);
    voxelGrid.initWithoutData3D(gridWidth, gridHeight, gridDepth, GL_RGBA32F);
    voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

    Shader scatterLightShader("scatterLight.comp", GL_COMPUTE_SHADER, BufferBindings::g_definitions);
    ShaderProgram sp({ scatterLightShader });

    Shader accumShader("accumulateVoxels.comp", GL_COMPUTE_SHADER, BufferBindings::g_definitions);
    ShaderProgram accumSp({ accumShader });

    auto u_voxelGridImg = std::make_shared<Uniform<GLuint64>>("voxelGrid", voxelGrid.generateImageHandle(GL_RGBA32F));
    sp.addUniform(u_voxelGridImg);
    accumSp.addUniform(u_voxelGridImg);

    auto u_gridDim = std::make_shared<Uniform<glm::ivec3>>("gridDim", glm::ivec3(gridWidth, gridHeight, gridDepth));
    sp.addUniform(u_gridDim);
    accumSp.addUniform(u_gridDim);

    auto u_debugMode = std::make_shared<Uniform<int>>("debugMode", 2);
    sp.addUniform(u_debugMode);

    auto u_maxRange = std::make_shared<Uniform<float>>("maxRange", 10.0f);
    sp.addUniform(u_maxRange);

    Pilotview playerCamera(screenWidth, screenHeight);
    const glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), screenWidth / static_cast<float>(screenHeight), screenNear, screenFar);

    Buffer matrixSSBO(GL_SHADER_STORAGE_BUFFER);
    matrixSSBO.setStorage(std::array<PlayerCameraInfo, 1>{{playerCamera.getView(), playerProj, playerCamera.getPosition()}}, GL_DYNAMIC_STORAGE_BIT);
    matrixSSBO.bindBase(BufferBindings::Binding::cameraParameters);

    FogInfo fog = { glm::vec3(1.0f), 0.5f, 10.f, 10.f };
    Buffer fogSSBO(GL_SHADER_STORAGE_BUFFER);
    fogSSBO.setStorage(std::array<FogInfo, 1>{ fog }, GL_DYNAMIC_STORAGE_BIT);
    fogSSBO.bindBase(static_cast<BufferBindings::Binding>(2));

    SimplexNoise noise;
    noise.bindNoiseBuffer(static_cast<BufferBindings::Binding>(3));

    auto l1 = std::make_shared<Light>(glm::vec3(1.0f), glm::vec3(1.0f, -1.0f, 1.0f));
    l1->setPosition({ 0.0f, 10.0f, 0.0f }); // position for shadow map only
    l1->recalculateLightSpaceMatrix();
    LightManager lm;
    lm.addLight(l1);
    lm.uploadLightsToGPU();

    VoxelDebugRenderer vdbgr({ gridWidth, gridHeight, gridDepth }, ScreenInfo{ screenWidth, screenHeight, screenNear, screenFar });
    glBindImageTexture(0, voxelGrid.getName(), 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA32F);

    Timer timer;
    int dbgcActive = 1;

    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window))
    {
        timer.start();
        
        glfwPollEvents();
        
        if (dbgcActive)
        {
            vdbgr.updateCamera(window);
        }
        else
        {
            playerCamera.update(window);
            matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
            matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));
        } 
         
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //lm.renderShadowMaps({}); //TODO: put scene in here

        voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

		noise.getNoiseBuffer().setContentSubData(static_cast<float>(glfwGetTime()), offsetof(GpuNoiseInfo, time));
        sp.use();
        glDispatchCompute(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(groupSize))),
            static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(groupSize))),
            static_cast<GLint>(std::ceil(gridDepth / static_cast<float>(groupSize))));
		//glDispatchComputeGroupSizeARB(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(groupSize))),
        //static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(groupSize))),
        //static_cast<GLint>(std::ceil(gridDepth / static_cast<float>(groupSize))), groupSize, groupSize, groupSize);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        //accumSp.use();
        //glDispatchCompute(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(8))),
        //    static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(8))), 1);

        //glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        vdbgr.draw();

        timer.stop();

		if constexpr (renderimgui)
		{	//imgui window
			ImGui_ImplGlfwGL3_NewFrame();
			static int tab = 0;
			ImGui::Begin("Main Window", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize);
			//Menu
			if (ImGui::BeginMenuBar())
			{
				if (ImGui::MenuItem("Density"))
					tab = 1;
				if (ImGui::MenuItem("Camera"))
					tab = 2;
				if (ImGui::MenuItem("Renderer"))
					tab = 3;
				if (ImGui::MenuItem("Light"))
					tab = 4;
				if (ImGui::MenuItem("Fog"))
					tab = 5;
				if (ImGui::MenuItem("Image"))
					tab = 6;
				ImGui::MenuItem("     ");
				//ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 20);
				if (ImGui::MenuItem("x"))
					tab = 0;
				ImGui::EndMenuBar();
			}
			//Body
			switch (tab) {
				//Density
			case 1:
			{
				ImGui::Text("Density and Noise Settings");
				ImGui::Separator();
				if (ImGui::SliderFloat("Noise Scale", &noise.m_noiseScale, 0.0f, 20.0f))
					noise.getNoiseBuffer().setContentSubData(noise.m_noiseScale, offsetof(GpuNoiseInfo, noiseScale));
				if (ImGui::SliderFloat("Noise Speed", &noise.m_noiseSpeed, 0.0f, 1.0f))
                    noise.getNoiseBuffer().setContentSubData(noise.m_noiseSpeed, offsetof(GpuNoiseInfo, noiseSpeed));
				if (ImGui::SliderFloat("Density Factor", &noise.m_densityFactor, 0.0f, 10.0f))
                    noise.getNoiseBuffer().setContentSubData(noise.m_densityFactor, offsetof(GpuNoiseInfo, heightDensityFactor));
				break;
			}
			//Camera
			case 2:
			{
				ImGui::Text("Camera Settings");
				ImGui::Separator();
				ImGui::RadioButton("Player Camera", &dbgcActive, 0); ImGui::SameLine();
				ImGui::RadioButton("Debug Camera", &dbgcActive, 1);
                ImGui::SliderFloat("Camera max range", &u_maxRange->getContentRef(), 0.0f, 100.0f);
				if (ImGui::Button("Reset Player Camera"))
				{
					playerCamera.reset();
					matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
					matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));
				}
				vdbgr.drawCameraGuiContent();
				break;
			}
			//Voxel debug renderer and shaders
			case 3:
			{
				vdbgr.drawGuiContent();
				sp.showReloadShaderGUIContent({ scatterLightShader }, "Voxel");
				//accumSp.showReloadShaderGUIContent({ accumShader }, "Accumulation");
				break;
			}
			//Light
			case 4:
			{
				ImGui::Text("Light Settings");
				lm.showLightGUIsContent();
				break;
			}
			//Fog
			case 5:
			{
				ImGui::Text("Fog Settings");
				if (ImGui::SliderFloat3("Albedo", value_ptr(fog.fogAlbedo), 0.0f, 1.0f))
					fogSSBO.setContentSubData(fog.fogAlbedo, offsetof(FogInfo, fogAlbedo));
				if (ImGui::SliderFloat("Anisotropy", &fog.fogAnisotropy, 0.0f, 1.0f))
					fogSSBO.setContentSubData(fog.fogAnisotropy, offsetof(FogInfo, fogAnisotropy));
				if (ImGui::SliderFloat("Scattering", &fog.fogScatteringCoeff, 0.0f, 100.0f))
					fogSSBO.setContentSubData(fog.fogScatteringCoeff, offsetof(FogInfo, fogScatteringCoeff));
				if (ImGui::SliderFloat("Absorption", &fog.fogAbsorptionCoeff, 0.0f, 100.0f))
					fogSSBO.setContentSubData(fog.fogAbsorptionCoeff, offsetof(FogInfo, fogAbsorptionCoeff));
				break;
			}

			case 6:
			{
				ImGui::Text("Image content settings");
				ImGui::RadioButton("Full volumetric values (outColor)", &u_debugMode->getContentRef(), 0);
				ImGui::RadioButton("worldPos, density", &u_debugMode->getContentRef(), 1);
				ImGui::RadioButton("worldPos, outColor.r", &u_debugMode->getContentRef(), 2);
				ImGui::RadioButton("lighting, density", &u_debugMode->getContentRef(), 3);
				break;
			}

			default:
				break;

			}
			if (tab) ImGui::Separator();
			timer.drawGuiContent(window);
			ImGui::End();
			ImGui::Render();
			ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
		}

        glfwSwapBuffers(window);
    }

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
