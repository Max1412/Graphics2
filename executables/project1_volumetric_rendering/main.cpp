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

#include "IO/GuiFont.cpp"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

constexpr int screenWidth = 1600;
constexpr int screenHeight = 900;
constexpr float screenNear = 0.1f;
constexpr float screenFar = 10000.f;
constexpr int gridWidth = 320;
constexpr int gridHeight = 180;
constexpr int gridDepth = 256;
constexpr int groupSize = 4;

constexpr bool renderimgui = true;

struct PlayerCameraInfo
{
    glm::mat4 playerViewMatrix;
    glm::mat4 playerProjMatrix;
    glm::vec3 camPos; float pad = 0.0f;
};

struct FogInfo
{
    glm::vec3 fogAlbedo;
    float fogAnisotropy;
    float fogScatteringCoeff;
    float fogAbsorptionCoeff;
    float fogDensity;
    float pad1 = 0.0f;
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
	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowBorderSize = 0.0f;
	style.PopupBorderSize = 0.0f;
	style.WindowRounding = 2.0f;
	ImGuiIO& io = ImGui::GetIO();
	ImFont* font = io.Fonts->AddFontFromMemoryCompressedBase85TTF(GuiFont_compressed_data_base85, 16.0f);
	ImFont* fontMono = io.Fonts->AddFontDefault();
	fontMono->DisplayOffset.y = 2.0f;
	ImVec4* colors = ImGui::GetStyle().Colors;
	colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 0.94f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);

    int curScene = 0;
    std::array<const char*, 3> scenes = { "Sponza", "Breakfast Room", "San Miguel" };

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
    int fxaaIterations = 8;
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

    auto u_maxRange = std::make_shared<Uniform<float>>("maxRange", sceneParams.at(curScene).maxRange);
    sp.addUniform(u_maxRange);

    Pilotview playerCamera(screenWidth, screenHeight);
    const glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), screenWidth / static_cast<float>(screenHeight), screenNear, screenFar);

    Buffer matrixSSBO(GL_SHADER_STORAGE_BUFFER);
    matrixSSBO.setStorage(std::array<PlayerCameraInfo, 1>( {playerCamera.getView(), playerProj, playerCamera.getPosition(), 0} ), GL_DYNAMIC_STORAGE_BIT);
    matrixSSBO.bindBase(BufferBindings::Binding::cameraParameters);

	FogInfo fog = {
	    sceneParams.at(curScene).fog.albedo, sceneParams.at(curScene).fog.anisotropy,
	    sceneParams.at(curScene).fog.scattering, sceneParams.at(curScene).fog.absorption,
	    sceneParams.at(curScene).fog.density
	};
    Buffer fogSSBO(GL_SHADER_STORAGE_BUFFER);
    fogSSBO.setStorage(std::array<FogInfo, 1>{ fog }, GL_DYNAMIC_STORAGE_BIT);
    fogSSBO.bindBase(static_cast<BufferBindings::Binding>(2));

    SimplexNoise sponzaNoise(sceneParams.at(0).noise.scale, sceneParams.at(0).noise.speed, sceneParams.at(0).noise.densityFactor, sceneParams.at(0).noise.densityHeight);
	SimplexNoise breakfastNoise(sceneParams.at(1).noise.scale, sceneParams.at(1).noise.speed, sceneParams.at(1).noise.densityFactor, sceneParams.at(1).noise.densityHeight);
	sponzaNoise.bindNoiseBuffer(static_cast<BufferBindings::Binding>(3));

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

	std::vector<std::shared_ptr<ModelImporter>> sceneVec = {
	    std::make_shared<ModelImporter>("sponza/sponza.obj"),
	    std::make_shared<ModelImporter>("breakfast_room/breakfast_room.obj"),
	    std::make_shared<ModelImporter>("San_Miguel/san-miguel.obj")
	};

    // not needed for multidraw
	// sceneVec.at(0)->registerUniforms(modelSp);
    // sceneVec.at(1)->registerUniforms(modelSp);
    // sceneVec.at(2)->registerUniforms(modelSp);

	// lights (parameters intended for sponza)
	std::vector<LightManager> lightMngrVec(3);

	// SPONZA LIGHTS
	for (unsigned int i = 0; i < sceneParams.at(0).lights.size(); i++)
	{
		// spot light
		if (sceneParams.at(0).lights[i].constant && sceneParams.at(0).lights[i].cutOff)
		{
			auto spot = std::make_shared<Light>(sceneParams.at(0).lights[i].color, sceneParams.at(0).lights[i].position, sceneParams.at(0).lights[i].direction,
                sceneParams.at(0).lights[i].constant, sceneParams.at(0).lights[i].linear, sceneParams.at(0).lights[i].quadratic, sceneParams.at(0).lights[i].cutOff, sceneParams.at(0).lights[i].outerCutOff);
			spot->setPCFKernelSize(sceneParams.at(0).lights[i].pcfKernelSize);
            lightMngrVec.at(0).addLight(spot);
		}
		// point light
		else if (sceneParams.at(0).lights[i].constant)
		{
			auto point = std::make_shared<Light>(sceneParams.at(0).lights[i].color, sceneParams.at(0).lights[i].position, sceneParams.at(0).lights[i].constant, sceneParams.at(0).lights[i].linear, sceneParams.at(0).lights[i].quadratic);
			point->setPCFKernelSize(sceneParams.at(0).lights[i].pcfKernelSize);
            lightMngrVec.at(0).addLight(point);
		}
		// directional light
		else
		{
			auto directional = std::make_shared<Light>(sceneParams.at(0).lights[i].color, sceneParams.at(0).lights[i].direction);
            lightMngrVec.at(0).addLight(directional);
		}
	}
    lightMngrVec.at(0).setOuterSceneBoundingBoxToAllLights(sceneVec.at(0)->getOuterBoundingBox());
    lightMngrVec.at(0).uploadLightsToGPU();

    // BREAKFAST ROOM LIGHTS
    for (unsigned int i = 0; i < sceneParams.at(1).lights.size(); i++)
    {
        // spot light
        if (sceneParams.at(1).lights[i].constant && sceneParams.at(1).lights[i].cutOff)
        {
            auto spot = std::make_shared<Light>(sceneParams.at(1).lights[i].color, sceneParams.at(1).lights[i].position, sceneParams.at(1).lights[i].direction,
                sceneParams.at(1).lights[i].constant, sceneParams.at(1).lights[i].linear, sceneParams.at(1).lights[i].quadratic, sceneParams.at(1).lights[i].cutOff, sceneParams.at(1).lights[i].outerCutOff);
            spot->setPCFKernelSize(sceneParams.at(1).lights[i].pcfKernelSize);
            lightMngrVec.at(1).addLight(spot);
        }
        // point light
        else if (sceneParams.at(curScene).lights[i].constant)
        {
            auto point = std::make_shared<Light>(sceneParams.at(1).lights[i].color, sceneParams.at(1).lights[i].position, sceneParams.at(1).lights[i].constant, sceneParams.at(1).lights[i].linear, sceneParams.at(1).lights[i].quadratic);
            point->setPCFKernelSize(sceneParams.at(1).lights[i].pcfKernelSize);
            lightMngrVec.at(1).addLight(point);
        }
        // directional light
        else
        {
            auto directional = std::make_shared<Light>(sceneParams.at(1).lights[i].color, sceneParams.at(1).lights[i].direction);
            lightMngrVec.at(1).addLight(directional);
        }
    }
    lightMngrVec.at(1).setOuterSceneBoundingBoxToAllLights(sceneVec.at(1)->getOuterBoundingBox());
    lightMngrVec.at(1).uploadLightsToGPU();
	
    // SAN MIGUEL LIGHTS
    for (unsigned int i = 0; i < sceneParams.at(2).lights.size(); i++)
    {
        // spot light
        if (sceneParams.at(2).lights[i].constant && sceneParams.at(2).lights[i].cutOff)
        {
            auto spot = std::make_shared<Light>(sceneParams.at(2).lights[i].color, sceneParams.at(2).lights[i].position, sceneParams.at(2).lights[i].direction,
                sceneParams.at(2).lights[i].constant, sceneParams.at(2).lights[i].linear, sceneParams.at(2).lights[i].quadratic, sceneParams.at(2).lights[i].cutOff, sceneParams.at(2).lights[i].outerCutOff);
            spot->setPCFKernelSize(sceneParams.at(2).lights[i].pcfKernelSize);
            lightMngrVec.at(2).addLight(spot);
        }
        // point light
        else if (sceneParams.at(curScene).lights[i].constant)
        {
            auto point = std::make_shared<Light>(sceneParams.at(2).lights[i].color, sceneParams.at(2).lights[i].position, sceneParams.at(2).lights[i].constant, sceneParams.at(2).lights[i].linear, sceneParams.at(2).lights[i].quadratic);
            point->setPCFKernelSize(sceneParams.at(2).lights[i].pcfKernelSize);
            lightMngrVec.at(2).addLight(point);
        }
        // directional light
        else
        {
            auto directional = std::make_shared<Light>(sceneParams.at(2).lights[i].color, sceneParams.at(2).lights[i].direction);
            lightMngrVec.at(2).addLight(directional);
        }
    }
    lightMngrVec.at(2).setOuterSceneBoundingBoxToAllLights(sceneVec.at(2)->getOuterBoundingBox());
    lightMngrVec.at(2).uploadLightsToGPU();
	
	// set the active scene
    lightMngrVec.at(curScene).bindLightBuffer();
	sceneVec.at(curScene)->bindGPUbuffers();

	// set camera to starting pos and dir
	playerCamera.setPosition(sceneParams.at(curScene).cameraPos);
	playerCamera.setTheta(sceneParams.at(curScene).theta);
	playerCamera.setPhi(sceneParams.at(curScene).phi);
    playerCamera.setSensitivityFromBBox(sceneVec.at(curScene)->getOuterBoundingBox());

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

        playerCamera.update(window);
        matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
        matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));

        lightMngrVec.at(curScene).renderShadowMapsCulled(*sceneVec.at(curScene));

        // render to fbo
        hdrFBO.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

        sponzaNoise.getNoiseBuffer().setContentSubData(static_cast<float>(glfwGetTime()), offsetof(GpuNoiseInfo, time));
		breakfastNoise.getNoiseBuffer().setContentSubData(static_cast<float>(glfwGetTime()), offsetof(GpuNoiseInfo, time));

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

        // render skybox first
        glDepthMask(GL_FALSE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        skyboxSP.use();
        cube.draw();
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        sceneVec.at(curScene)->multiDrawCulled(modelSp, playerProj * playerCamera.getView()); //modelLoader.multiDraw(modelSp);

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
            //Menu
			if (ImGui::BeginMainMenuBar())
			{
				if (ImGui::MenuItem("Light"))
					tab = tab == 4 ? 0 : 4;
				if (ImGui::MenuItem("Density"))
					tab = tab == 1 ? 0 : 1;
				if (ImGui::MenuItem("Fog"))
					tab = tab == 5 ? 0 : 5;
				if (ImGui::MenuItem("Camera"))
					tab = tab == 2 ? 0 : 2;
				if (ImGui::MenuItem("Shader"))
					tab = tab == 3 ? 0 : 3;
				if (ImGui::MenuItem("FBO"))
					tab = tab == 6 ? 0 : 6;
				if (ImGui::BeginMenu("Scene"))
				{
					//list all scenes to select
					for (int i = 0; i < static_cast<int>(scenes.size()); i++)
					{
						if (ImGui::Selectable(scenes[i], curScene == i))
						{
							curScene = i;
							// bind active GPU light buffer
							lightMngrVec.at(curScene).bindLightBuffer();

							// bind active buffers from the scene
							sceneVec.at(curScene)->bindGPUbuffers();

							// reset camera and upload it to gpu
							playerCamera.setPosition(sceneParams.at(curScene).cameraPos);
							playerCamera.setTheta(sceneParams.at(curScene).theta);
							playerCamera.setPhi(sceneParams.at(curScene).phi);
							playerCamera.setSensitivityFromBBox(sceneVec.at(curScene)->getOuterBoundingBox());

							u_maxRange->setContent(sceneParams.at(curScene).maxRange);

							fog = { sceneParams.at(curScene).fog.albedo, sceneParams.at(curScene).fog.anisotropy, sceneParams.at(curScene).fog.scattering, sceneParams.at(curScene).fog.absorption, sceneParams.at(curScene).fog.density };
							fogSSBO.setContentSubData(fog, 0);

							if (curScene == 1)
							{
								breakfastNoise.bindNoiseBuffer(static_cast<BufferBindings::Binding>(3));
							}
							else if (curScene == 0)
							{
								sponzaNoise.bindNoiseBuffer(static_cast<BufferBindings::Binding>(3));
							}
							else if (curScene == 2)
							{
								// use breakfast room noise here too for now
								breakfastNoise.bindNoiseBuffer(static_cast<BufferBindings::Binding>(3));
								playerCamera.reset();
							}
							else
							{
								playerCamera.reset();
							}

							matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
							matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));
						}
					}
					ImGui::EndMenu();
				}

				//change to monospaced font for timer
				ImGui::PushFont(fontMono);
				colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
				timer.drawGuiContent(window, true);
				colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.29f, 0.48f, 0.54f);
				ImGui::PopFont();

				ImGui::EndMainMenuBar();
			}
            //Body
			if (tab)
				ImGui::Begin("Main Window", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize);
            switch (tab) {
            //Density
            case 1:
            {
				if(curScene == 0)
					sponzaNoise.showNoiseGUIContent();
				else if (curScene == 1)
					breakfastNoise.showNoiseGUIContent();
                if(curScene == 2)
                    breakfastNoise.showNoiseGUIContent();
                break;
            }
            //Camera
            case 2:
            {
                ImGui::Text("Camera Settings");
                ImGui::Separator();
				ImGui::Text("Camera Position: (%.1f,%.1f,%.1f)", playerCamera.getPosition().x, playerCamera.getPosition().y, playerCamera.getPosition().z);
                ImGui::SliderFloat("Camera max voxel range", &u_maxRange->getContentRef(), 10.0f, screenFar);
                if (ImGui::Button("Reset Player Camera"))
                {
                    playerCamera.reset();
                    matrixSSBO.setContentSubData(playerCamera.getView(), offsetof(PlayerCameraInfo, playerViewMatrix));
                    matrixSSBO.setContentSubData(playerCamera.getPosition(), offsetof(PlayerCameraInfo, camPos));
				}
                break;
            }
            //Shader GUIs
            case 3:
            {
                sp.showReloadShaderGUIContent({ scatterLightShader }, "Voxel");
                accumSp.showReloadShaderGUIContent({ accumShader }, "Accumulation");
				modelSp.showReloadShaderGUIContent({ modelVertexShader, modelFragmentShader }, "Forward Rendering");
				fboHDRtoLDRSP.showReloadShaderGUIContent({ fboVS, fboHDRtoLDRFS }, "FBO: HDR to LDR");
                break;
            }
            //Light
            case 4:
            {
                ImGui::Text("Light Settings");
				lightMngrVec.at(curScene).showLightGUIsContent();
                break;
            }
            //Fog
            case 5:
            {
                ImGui::Text("Fog Settings");
				ImGui::Separator();
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
                ImGui::Text("FBO settings");
				ImGui::Separator();
                if (ImGui::SliderFloat("Gamma", &gamma, 0.0f, 5.0f))
                    u_gamma->setContent(gamma);
                if (ImGui::SliderFloat("Exposure", &exposure, 0.0f, 1.0f))
                    u_exposure->setContent(exposure);
                if (ImGui::SliderInt("FXAA iterations", &fxaaIterations, 0, 8))
                    u_fxaaIterations->setContent(fxaaIterations);
                break;
            }

            default:
                break;

            }
			if (tab) {
				ImGui::SetWindowPos(ImVec2(0, 22), ImGuiCond_Once);
				ImGui::End();
			}
            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(window);
    }

    ImGui_ImplGlfwGL3_Shutdown();

    // close window
    glfwDestroyWindow(window);
}
