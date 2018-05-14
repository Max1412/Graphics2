#include <glbinding/gl/gl.h>
#include "Rendering/Light.h"
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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "Rendering/FrameBuffer.h"

constexpr int screenWidth = 1600;
constexpr int screenHeight = 900;
constexpr float screenNear = 0.1f;
constexpr float screenFar = 1000.f;
constexpr int gridWidth = screenWidth / 10;
constexpr int gridHeight = screenHeight / 10;
constexpr int gridDepth = static_cast<int>(screenFar) / 100;
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

struct NoiseInfo
{
	GLuint64 permTexture;
	GLuint64 simplexTexture;
	GLuint64 gradientTexture;
	float time;
	float heightDensityFactor;
	float noiseScale;
	float noiseSpeed;
};

constexpr int perm[256] = { 151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };
//gradients for 4d noise, midpoints of each edge of a tesseract
constexpr int gradient4[32][4] = { { 0,1,1,1 },{ 0,1,1,-1 },{ 0,1,-1,1 },{ 0,1,-1,-1 }, // 32 tesseract edges
{ 0,-1,1,1 },{ 0,-1,1,-1 },{ 0,-1,-1,1 },{ 0,-1,-1,-1 },
{ 1,0,1,1 },{ 1,0,1,-1 },{ 1,0,-1,1 },{ 1,0,-1,-1 },
{ -1,0,1,1 },{ -1,0,1,-1 },{ -1,0,-1,1 },{ -1,0,-1,-1 },
{ 1,1,0,1 },{ 1,1,0,-1 },{ 1,-1,0,1 },{ 1,-1,0,-1 },
{ -1,1,0,1 },{ -1,1,0,-1 },{ -1,-1,0,1 },{ -1,-1,0,-1 },
{ 1,1,1,0 },{ 1,1,-1,0 },{ 1,-1,1,0 },{ 1,-1,-1,0 },
{ -1,1,1,0 },{ -1,1,-1,0 },{ -1,-1,1,0 },{ -1,-1,-1,0 } };
//simplex lookup table
constexpr unsigned char simplex4[][4] = { { 0,64,128,192 },{ 0,64,192,128 },{ 0,0,0,0 },
{ 0,128,192,64 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },{ 64,128,192,0 },
{ 0,128,64,192 },{ 0,0,0,0 },{ 0,192,64,128 },{ 0,192,128,64 },
{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },{ 64,192,128,0 },
{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 64,128,0,192 },{ 0,0,0,0 },{ 64,192,0,128 },{ 0,0,0,0 },
{ 0,0,0,0 },{ 0,0,0,0 },{ 128,192,0,64 },{ 128,192,64,0 },
{ 64,0,128,192 },{ 64,0,192,128 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 0,0,0,0 },{ 128,0,192,64 },{ 0,0,0,0 },{ 128,64,192,0 },
{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 128,0,64,192 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 192,0,64,128 },{ 192,0,128,64 },{ 0,0,0,0 },{ 192,64,128,0 },
{ 128,64,0,192 },{ 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 },
{ 192,64,0,128 },{ 0,0,0,0 },{ 192,128,0,64 },{ 192,128,64,0 } };

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
    GLuint64 handle = voxelGrid.generateImageHandle(GL_RGBA32F);
    voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

    Buffer imageHoldingSSBO(GL_SHADER_STORAGE_BUFFER);
    imageHoldingSSBO.setStorage(std::vector<GLuint64>{ handle }, GL_DYNAMIC_STORAGE_BIT);
    imageHoldingSSBO.bindBase(static_cast<BufferBindings::Binding>(0));

    Shader scatterLightShader("scatterLight.comp", GL_COMPUTE_SHADER, BufferBindings::g_definitions);
    ShaderProgram sp({ scatterLightShader });

    Shader accumShader("accumulateVoxels.comp", GL_COMPUTE_SHADER, BufferBindings::g_definitions);
    ShaderProgram accumSp({ accumShader });

    auto u_gridDim = std::make_shared<Uniform<glm::ivec3>>("gridDim", glm::ivec3(gridWidth, gridHeight, gridDepth));
    sp.addUniform(u_gridDim);
    accumSp.addUniform(u_gridDim);

    auto u_debugMode = std::make_shared<Uniform<int>>("debugMode", 2);
    sp.addUniform(u_debugMode);

	//set variables and uniforms for noise and density
	float time = (float)glfwGetTime();
	float densityFactor = 4.0f;
	float noiseScale = 6.0f;
	float noiseSpeed = 0.1f;
	GLuint64 permTextureID, simplexTextureID, gradientTextureID;

	//creates 2d texture combining permutation and gradient lookup table
	Texture permTexture(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST);
	{
		char *pixels;
		pixels = (char*)malloc(256 * 256 * 4);
		for (int i = 0; i<256; i++)
			for (int j = 0; j<256; j++) {
				int offset = (i * 256 + j) * 4;
				char value = perm[(j + perm[i]) & 0xFF];
				pixels[offset] = gradient4[2 * (value & 0x0F)][0] * 64 + 64;   // Gradient x
				pixels[offset + 1] = gradient4[2 * (value & 0x0F)][1] * 64 + 64; // Gradient y
				pixels[offset + 2] = gradient4[2 * (value & 0x0F)][2] * 64 + 64; // Gradient z
				pixels[offset + 3] = value;                     // Permuted index
			}

		permTexture.initWithData2D(pixels, 256, 256);
		permTextureID = permTexture.generateHandle();
	}

	//create 1d texture for simplex traversal lookup
	Texture simplexTexture(GL_TEXTURE_1D, GL_NEAREST, GL_NEAREST);
	simplexTexture.initWithData1D(simplex4, 64);
	simplexTextureID = simplexTexture.generateHandle();

	//create 2d gradient lookup table
	Texture gradientTexture(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST);
	{
		char *pixels;
		pixels = (char*)malloc(256 * 256 * 4);
		for (int i = 0; i<256; i++)
			for (int j = 0; j<256; j++) {
				int offset = (i * 256 + j) * 4;
				char value = perm[(j + perm[i]) & 0xFF];
				pixels[offset] = gradient4[value & 0x1F][0] * 64 + 64;   // Gradient x
				pixels[offset + 1] = gradient4[value & 0x1F][1] * 64 + 64; // Gradient y
				pixels[offset + 2] = gradient4[value & 0x1F][2] * 64 + 64; // Gradient z
				pixels[offset + 3] = gradient4[value & 0x1F][3] * 64 + 64; // Gradient z
			}

		gradientTexture.initWithData2D(pixels, 256, 256);
		gradientTextureID = gradientTexture.generateHandle();
	}

    Pilotview playerCamera(screenWidth, screenHeight);
    glm::mat4 playerProj = glm::perspective(glm::radians(60.0f), screenWidth / static_cast<float>(screenHeight), screenNear, screenFar);

    Buffer matrixSSBO(GL_SHADER_STORAGE_BUFFER);
    matrixSSBO.setStorage(std::array<PlayerCameraInfo, 1>{{playerCamera.getView(), playerProj, playerCamera.getPosition()}}, GL_DYNAMIC_STORAGE_BIT);
    matrixSSBO.bindBase(static_cast<BufferBindings::Binding>(1));

    FogInfo fog = { glm::vec3(1.0f), 0.5f, 10.f, 10.f };
    Buffer fogSSBO(GL_SHADER_STORAGE_BUFFER);
    fogSSBO.setStorage(std::array<FogInfo, 1>{ fog }, GL_DYNAMIC_STORAGE_BIT);
    fogSSBO.bindBase(static_cast<BufferBindings::Binding>(2));

	Buffer noiseSSBO(GL_SHADER_STORAGE_BUFFER);
	noiseSSBO.setStorage(std::array<NoiseInfo, 1>{ {permTextureID, simplexTextureID, gradientTextureID, time, densityFactor, noiseScale, noiseSpeed}}, GL_DYNAMIC_STORAGE_BIT);
	noiseSSBO.bindBase(static_cast<BufferBindings::Binding>(3));

    auto l1 = std::make_shared<Light>(glm::vec3(1.0f), glm::vec3(1.0f, -1.0f, 1.0f));
    l1->setPosition({ 0.0f, 10.0f, 0.0f }); // position for shadow map only
    l1->recalculateLightSpaceMatrix();
    LightManager lm;
    lm.addLight(l1);
    lm.uploadLightsToGPU();

    VoxelDebugRenderer vdbgr({ gridWidth, gridHeight, gridDepth }, ScreenInfo{ screenWidth, screenHeight, screenNear, screenFar });

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

        lm.renderShadowMaps({}); //TODO: put scene in here

        voxelGrid.clearTexture(GL_RGBA, GL_FLOAT, glm::vec4(-1.0f), 0);

		noiseSSBO.setContentSubData(static_cast<float>(glfwGetTime()), offsetof(NoiseInfo, time));
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
				if (ImGui::SliderFloat("Noise Scale", &noiseScale, 0.0f, 20.0f))
					noiseSSBO.setContentSubData(noiseScale, offsetof(NoiseInfo, noiseScale));
				if (ImGui::SliderFloat("Noise Speed", &noiseSpeed, 0.0f, 1.0f))
					noiseSSBO.setContentSubData(noiseSpeed, offsetof(NoiseInfo, noiseSpeed));
				if (ImGui::SliderFloat("Density Factor", &densityFactor, 0.0f, 10.0f))
					noiseSSBO.setContentSubData(densityFactor, offsetof(NoiseInfo, heightDensityFactor));
				break;
			}
			//Camera
			case 2:
			{
				ImGui::Text("Camera Settings");
				ImGui::Separator();
				ImGui::RadioButton("Player Camera", &dbgcActive, 0); ImGui::SameLine();
				ImGui::RadioButton("Debug Camera", &dbgcActive, 1);
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
