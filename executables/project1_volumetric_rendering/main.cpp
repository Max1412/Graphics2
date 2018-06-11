#include <glbinding/gl/gl.h>
#include "Rendering/Light.h"
#include "Rendering/SimplexNoise.h"
#include "Rendering/Quad.h"
#include "Rendering/Cubemap.h"
#include "Rendering/SkyBoxCube.h"
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
#include "Rendering/VoxelDebugRenderer.h"
#include "Rendering/Pilotview.h"
#include "Rendering/LightManager.h"
#include "Rendering/Parameters.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

constexpr int screenWidth = 1600;
constexpr int screenHeight = 900;
constexpr float screenNear = 0.1f;
constexpr float screenFar = 10000.f;
constexpr int gridWidth = 190;
constexpr int gridHeight = 90;
constexpr int gridDepth = 64;
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
    float fogDensity;
    float pad1;
};

int main()
{
    // init glfw, open window, manage context
    GLFWwindow* window = util::setupGLFWwindow(screenWidth, screenHeight, "Volumetric Lighting/Fog");
    glfwSwapInterval(0);
    // init opengl
    util::initGL();

    // print OpenGL info
    util::printOpenGLInfo();

    util::enableDebugCallback();

    // set up imgui
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    // F B O : H D R -> L D R
    std::vector<Texture> hdrTex(1);
    hdrTex[0].initWithoutData(screenWidth, screenHeight, GL_RGBA32F);
    FrameBuffer hdrFBO(hdrTex);

    Shader fboVS("texSFQ.vert", GL_VERTEX_SHADER);
    Shader fboHDRtoLDRFS("HDRtoLDR.frag", GL_FRAGMENT_SHADER);
    ShaderProgram fboHDRtoLDRSP(fboVS, fboHDRtoLDRFS);
    float exposure = 0.1f, gamma = 2.2f;
    auto u_exposure = std::make_shared<Uniform<float>>("exposure", exposure);
    auto u_gamma = std::make_shared<Uniform<float>>("gamma", gamma);
    auto u_hdrTexHandle = std::make_shared<Uniform<GLuint64>>("inputTexture", hdrTex[0].generateHandle());
    fboHDRtoLDRSP.addUniform(u_exposure);
    fboHDRtoLDRSP.addUniform(u_gamma);
    fboHDRtoLDRSP.addUniform(u_hdrTexHandle);

    // F B O : F X A A
    std::vector<Texture> fxaaTex(1);
    fxaaTex[0].setMinMagFilter(GL_LINEAR, GL_LINEAR);
    fxaaTex[0].setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
    fxaaTex[0].initWithoutData(screenWidth, screenHeight, GL_RGBA32F);
    FrameBuffer fxaaFBO(fxaaTex);

    Shader fxaaShader("fxaa.frag", GL_FRAGMENT_SHADER);
    ShaderProgram fxaaSP(fboVS, fxaaShader);
    int fxaaIterations = 2;
    auto u_fxaaIterations = std::make_shared<Uniform<int>>("iterations", fxaaIterations);
    auto u_fxaaTexHandle = std::make_shared<Uniform<GLuint64>>("screenTexture", fxaaTex[0].generateHandle());
    fxaaSP.addUniform(u_fxaaIterations);
    fxaaSP.addUniform(u_fxaaTexHandle);

    // S F Q
    Quad fboQuad;

	// V O L U M E T R I C

    Image voxelGrid(GL_TEXTURE_3D, GL_LINEAR, GL_LINEAR);
    voxelGrid.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
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

    auto u_debugMode = std::make_shared<Uniform<int>>("debugMode", 0);
    sp.addUniform(u_debugMode);

    auto u_maxRange = std::make_shared<Uniform<float>>("maxRange", sponza.maxRange);
    sp.addUniform(u_maxRange);

    Pilotview playerCamera(screenWidth, screenHeight);
    glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), screenWidth / static_cast<float>(screenHeight), screenNear, screenFar);

    Buffer matrixSSBO(GL_SHADER_STORAGE_BUFFER);
    matrixSSBO.setStorage(std::array<PlayerCameraInfo, 1>{ {playerCamera.getView(), playerProj, playerCamera.getPosition()}}, GL_DYNAMIC_STORAGE_BIT);
    matrixSSBO.bindBase(BufferBindings::Binding::cameraParameters);

	FogInfo fog = { sponza.fog.albedo, sponza.fog.anisotropy, sponza.fog.scattering, sponza.fog.absorption, sponza.fog.density };
    Buffer fogSSBO(GL_SHADER_STORAGE_BUFFER);
    fogSSBO.setStorage(std::array<FogInfo, 1>{ fog }, GL_DYNAMIC_STORAGE_BIT);
    fogSSBO.bindBase(static_cast<BufferBindings::Binding>(2));

    SimplexNoise noise(sponza.noise.scale, sponza.noise.speed, sponza.noise.densityFactor, sponza.noise.densityHeight);
    noise.bindNoiseBuffer(static_cast<BufferBindings::Binding>(3));

    VoxelDebugRenderer vdbgr({ gridWidth, gridHeight, gridDepth }, ScreenInfo{ screenWidth, screenHeight, screenNear, screenFar });
    glBindImageTexture(0, voxelGrid.getName(), 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA32F);

    // skybox stuff
    const Shader skyboxVS("cubemap2.vert", GL_VERTEX_SHADER, BufferBindings::g_definitions);
    const Shader skyboxFS("cubemap2.frag", GL_FRAGMENT_SHADER, BufferBindings::g_definitions);
    ShaderProgram skyboxSP(skyboxVS, skyboxFS);

    SkyBoxCube cube;

    Cubemap cubemapSkybox;
    cubemapSkybox.loadFromFile(util::gs_resourcesPath / "/skybox/skybox.jpg");
    cubemapSkybox.generateHandle();

    auto u_skyboxTexHandle = std::make_shared<Uniform<GLuint64>>("skybox", cubemapSkybox.getHandle());
    skyboxSP.addUniform(u_skyboxTexHandle);

    Timer timer;
    int dbgcActive = 0;
	bool dbgrndr = false;

	// R E N D E R I N G

	Shader modelVertexShader("modelVertVolumetricMD.vert", GL_VERTEX_SHADER, BufferBindings::g_definitions);
	Shader modelFragmentShader("modelFragVolumetricMD.frag", GL_FRAGMENT_SHADER, BufferBindings::g_definitions);
	ShaderProgram modelSp(modelVertexShader, modelFragmentShader);

    auto u_voxelGridTex = std::make_shared<Uniform<GLuint64>>("voxelGrid", voxelGrid.generateHandle());
    auto u_screenRes = std::make_shared<Uniform<glm::vec2>>("screenRes", glm::vec2(screenWidth, screenHeight));

	modelSp.addUniform(u_maxRange);
    modelSp.addUniform(u_voxelGridTex);
    modelSp.addUniform(u_screenRes);
    skyboxSP.addUniform(u_maxRange);
    skyboxSP.addUniform(u_voxelGridTex);
    skyboxSP.addUniform(u_screenRes);

    int activeScene = 0;
    std::array<const char*, 2> scenes = { "Sponza", "Breakfast Room" };

	std::vector<std::shared_ptr<ModelImporter>> sceneVec = { std::make_shared<ModelImporter>("sponza/sponza.obj"), std::make_shared<ModelImporter>("breakfast_room/breakfast_room.obj") };
	//ModelImporter modelLoader("sponza/sponza.obj");
	sceneVec.at(0)->registerUniforms(modelSp);
    sceneVec.at(1)->registerUniforms(modelSp);

	// lights (parameters intended for sponza)
	std::vector<LightManager> lightMngrVec(2);

	// SPONZA LIGHTS
	for (unsigned int i = 0; i < sponza.lights.size(); i++)
	{
		// spot light
		if (sponza.lights[i].constant && sponza.lights[i].cutOff)
		{
			auto spot = std::make_shared<Light>(sponza.lights[i].color, sponza.lights[i].position, sponza.lights[i].direction, sponza.lights[i].constant, sponza.lights[i].linear, sponza.lights[i].quadratic, sponza.lights[i].cutOff, sponza.lights[i].outerCutOff);
			spot->setPCFKernelSize(sponza.lights[i].pcfKernelSize);
            lightMngrVec.at(0).addLight(spot);
		}
		// point light
		else if (sponza.lights[i].constant)
		{
			auto point = std::make_shared<Light>(sponza.lights[i].color, sponza.lights[i].position, sponza.lights[i].constant, sponza.lights[i].linear, sponza.lights[i].quadratic);
			point->setPCFKernelSize(sponza.lights[i].pcfKernelSize);
            lightMngrVec.at(0).addLight(point);
		}
		// directional light
		else
		{
			auto directional = std::make_shared<Light>(sponza.lights[i].color, sponza.lights[i].direction);
			directional->setPosition(sponza.lights[i].position); // position for shadow map only
			directional->recalculateLightSpaceMatrix();
            lightMngrVec.at(0).addLight(directional);
		}
	}

    lightMngrVec.at(0).uploadLightsToGPU();

    // BREAKFAST ROOM LIGHTS
    for (unsigned int i = 0; i < breakfast.lights.size(); i++)
    {
        // spot light
        if (breakfast.lights[i].constant && breakfast.lights[i].cutOff)
        {
            auto spot = std::make_shared<Light>(breakfast.lights[i].color, breakfast.lights[i].position, breakfast.lights[i].direction, breakfast.lights[i].constant, breakfast.lights[i].linear, breakfast.lights[i].quadratic, breakfast.lights[i].cutOff, breakfast.lights[i].outerCutOff);
            spot->setPCFKernelSize(breakfast.lights[i].pcfKernelSize);
            lightMngrVec.at(1).addLight(spot);
        }
        // point light
        else if (breakfast.lights[i].constant)
        {
            auto point = std::make_shared<Light>(breakfast.lights[i].color, breakfast.lights[i].position, breakfast.lights[i].constant, breakfast.lights[i].linear, breakfast.lights[i].quadratic);
            point->setPCFKernelSize(breakfast.lights[i].pcfKernelSize);
            lightMngrVec.at(1).addLight(point);
        }
        // directional light
        else
        {
            auto directional = std::make_shared<Light>(breakfast.lights[i].color, breakfast.lights[i].direction);
            directional->setPosition(breakfast.lights[i].position); // position for shadow map only
            directional->recalculateLightSpaceMatrix();
            lightMngrVec.at(1).addLight(directional);
        }
    }
    
    lightMngrVec.at(1).uploadLightsToGPU();

	// set the active scene
    lightMngrVec.at(activeScene).bindLightBuffer();
	sceneVec.at(activeScene)->bindGPUbuffers();

    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	while (!glfwWindowShouldClose(window))
    {
        timer.start();

        glfwPollEvents();

		vdbgr.updateCamera(window);

        if (dbgcActive)
        {
            //vdbgr.updateCamera(window);
        }
        else
        {
            playerCamera.update(window);
            matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
            matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));
		}
        lightMngrVec.at(activeScene).renderShadowMapsCulled(*sceneVec.at(activeScene));

        // render to fbo
        hdrFBO.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

        accumSp.use();
        glDispatchCompute(static_cast<GLint>(std::ceil(gridWidth / static_cast<float>(8))),
            static_cast<GLint>(std::ceil(gridHeight / static_cast<float>(8))), 1);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        //vdbgr.draw();

        if (!dbgrndr)
        {
            sceneVec.at(activeScene)->multiDrawCulled(modelSp, playerProj * playerCamera.getView()); //modelLoader.multiDraw(modelSp);

            // render skybox last
            glDepthFunc(GL_LEQUAL);
            glDisable(GL_CULL_FACE);
            skyboxSP.use();
            cube.draw();
            glEnable(GL_CULL_FACE);
            glDepthFunc(GL_LEQUAL);
        }

        // render to fxaa fbo now
        hdrFBO.unbind();
        fxaaFBO.bind();
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.4f, 0.1f, 0.1f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        fboHDRtoLDRSP.use();
        fboQuad.draw();
        glEnable(GL_DEPTH_TEST);

        // render to screen now
        fxaaFBO.unbind();
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.4f, 0.1f, 0.1f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        fxaaSP.use();
        fboQuad.draw();
        glEnable(GL_DEPTH_TEST);
        
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
                if (ImGui::MenuItem("FBO"))
                    tab = 7;
                if (ImGui::MenuItem("Scene"))
                    tab = 8;
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
                noise.showNoiseGUIContent();
                break;
            }
            //Camera
            case 2:
            {
                ImGui::Text("Camera Settings");
                ImGui::Separator();
				ImGui::Text("Camera Position: (%.1f,%.1f,%.1f)", playerCamera.getPosition().x, playerCamera.getPosition().y, playerCamera.getPosition().z);
                ImGui::RadioButton("Player Camera", &dbgcActive, 0); ImGui::SameLine();
                ImGui::RadioButton("Debug Camera", &dbgcActive, 1);
                ImGui::SliderFloat("Camera max voxel range", &u_maxRange->getContentRef(), 10.0f, screenFar);
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
                accumSp.showReloadShaderGUIContent({ accumShader }, "Accumulation");
				modelSp.showReloadShaderGUIContent({ modelVertexShader, modelFragmentShader }, "Forward Rendering");
				ImGui::Checkbox("Render Debug Renderer", &dbgrndr);
				if (dbgrndr)
				{
					dbgcActive = true;
					vdbgr.draw();
				}
                break;
            }
            //Light
            case 4:
            {
                ImGui::Text("Light Settings");
				lightMngrVec.at(activeScene).showLightGUIsContent();
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
                if (ImGui::SliderFloat("Scattering", &fog.fogScatteringCoeff, 0.0f, 1.0f))
                    fogSSBO.setContentSubData(fog.fogScatteringCoeff, offsetof(FogInfo, fogScatteringCoeff));
                if (ImGui::SliderFloat("Absorption", &fog.fogAbsorptionCoeff, 0.0f, 1.0f))
                    fogSSBO.setContentSubData(fog.fogAbsorptionCoeff, offsetof(FogInfo, fogAbsorptionCoeff));
                if (ImGui::SliderFloat("Density", &fog.fogDensity, 0.0f, 1.0f))
                    fogSSBO.setContentSubData(fog.fogDensity, offsetof(FogInfo, fogDensity));
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
            
            case 7:
            {
                ImGui::Text("FBO settings");
                fboHDRtoLDRSP.showReloadShaderGUIContent({fboVS, fboHDRtoLDRFS});
                if (ImGui::SliderFloat("Gamma", &gamma, 0.0f, 5.0f))
                    u_gamma->setContent(gamma);
                if (ImGui::SliderFloat("Exposure", &exposure, 0.0f, 1.0f))
                    u_exposure->setContent(exposure);
                if (ImGui::SliderInt("FXAA iterations", &fxaaIterations, 0, 8))
                    u_fxaaIterations->setContent(fxaaIterations);
                break;
            }

            case 8:
            {
                ImGui::Text("Scene selection");
                if(ImGui::Combo("Scenes", &activeScene, scenes.data(), scenes.size()))
                {
                    // bind active GPU light buffer
                    lightMngrVec.at(activeScene).bindLightBuffer();

                    // bind active buffers from the scene
                    sceneVec.at(activeScene)->bindGPUbuffers();

                    // reset camera and upload it to gpu
					playerCamera.reset();
					matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
					matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));
                }
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
