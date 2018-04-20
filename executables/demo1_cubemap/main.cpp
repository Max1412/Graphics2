#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
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
#include "Rendering/SkyBoxCube.h"
#include "Rendering/Cubemap.h"
#include "Rendering/Quad.h"
#include "Rendering/FrameBuffer.h"

const unsigned int width = 1600;
const unsigned int height = 900;

struct LightInfo
{
	glm::vec4 pos; //pos.w=0 dir., pos.w=1 point light
	glm::vec3 col;
	float spot_cutoff; //no spotlight if cutoff=0
	glm::vec3 spot_direction;
	float spot_exponent;
};

struct MaterialInfo
{
	glm::vec3 diffColor;
	float kd;
	glm::vec3 specColor;
	float ks;
	float shininess;
	float kt;
	int reflective;
	float pad2 = 0.0f;
};

struct FogInfo
{
	glm::vec3 col;
	float start;
	float end;
	float density;
	int mode;
	float pad = 0.0f;
};

int main()
{
	// init glfw, open window, manage context
	GLFWwindow* window = util::setupGLFWwindow(width, height, "Demo 1");
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

	// FBO stuff
	const Shader fboVS("texSFQ.vert", GL_VERTEX_SHADER);
	const Shader fboFS("postProcess.frag", GL_FRAGMENT_SHADER);
	ShaderProgram fboSP(fboVS, fboFS);
	fboSP.use();

	bool useGrayscale = false;
	auto grayscaleuniform = std::make_shared<Uniform<bool>>("useGrayscale", useGrayscale);
	bool useBlur = false;
	auto blurUniform = std::make_shared<Uniform<bool>>("useBlur", useBlur);
	bool useSharpen = false;
	auto sharpenUniform = std::make_shared<Uniform<bool>>("useSharpen", useSharpen);

	const auto dimUniform = std::make_shared<Uniform<glm::vec2>>("dimensions", glm::vec2(width, height));

	fboSP.addUniform(grayscaleuniform);
	fboSP.addUniform(dimUniform);
	fboSP.addUniform(blurUniform);
	fboSP.addUniform(sharpenUniform);

	Quad fboQuad;

	Texture fboTex;
	fboTex.initWithoutData(width, height, GL_RGBA8);
	fboTex.generateHandle();

	// put the texture handle into a SSBO
	const auto fboTexHandle = fboTex.getHandle();
	Buffer fboTexHandleBuffer(GL_SHADER_STORAGE_BUFFER);
	fboTexHandleBuffer.setStorage(std::array<GLuint64, 1>{fboTexHandle}, GL_DYNAMIC_STORAGE_BIT);
	fboTexHandleBuffer.bindBase(6);

	FrameBuffer fbo({fboTex});

	// skybox stuff
	const Shader skyboxVS("cubemap.vert", GL_VERTEX_SHADER);
	const Shader skyboxFS("cubemap.frag", GL_FRAGMENT_SHADER);
	ShaderProgram skyboxSP(skyboxVS, skyboxFS);
	SkyBoxCube cube;

	Cubemap CubemapSkybox;
	CubemapSkybox.loadFromFile(RESOURCES_PATH + std::string("/skybox/skybox.jpg"));
	CubemapSkybox.generateHandle();
	// put the texture handle into a SSBO
	Buffer textureHandleBuffer(GL_SHADER_STORAGE_BUFFER);
	textureHandleBuffer.setStorage(std::array<GLuint64, 1>{CubemapSkybox.getHandle()}, GL_DYNAMIC_STORAGE_BIT);
	textureHandleBuffer.bindBase(3);

	// actual stuff
	const Shader vs("demo1.vert", GL_VERTEX_SHADER);
	const Shader fs("demo1.frag", GL_FRAGMENT_SHADER);
	ShaderProgram sp(vs, fs);

	ModelImporter mi("bunny.obj");
	auto meshes = mi.getMeshes();
	auto bunny = meshes.at(0);

	// create a plane
	std::vector<glm::vec3> planePositions =
	{
		glm::vec3(-1, 0, -1), glm::vec3(-1, 0, 1), glm::vec3(1, 0, 1), glm::vec3(1, 0, -1)
	};

	std::vector<glm::vec3> planeNormals =
	{
		glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)
	};

	std::vector<unsigned> planeIndices =
	{
		0, 1, 2,
		2, 3, 0
	};

	Mesh plane(planePositions, planeNormals, planeIndices);

	sp.use();

	// create matrices for uniforms
	SimpleTrackball camera(width, height, 10.0f);
	glm::mat4 view = camera.getView();
	glm::vec3 cameraPos(camera.getPosition());

	glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / static_cast<float>(height), 0.1f, 1000.0f);
	glm::mat4 model(1.0f);
	model = translate(model, glm::vec3(0.0f, -1.0f, 0.0f));
	//model = glm::scale(model, glm::vec3(2.0f, 2.0f, 2.0f));
	bunny->setModelMatrix(model);

	glm::mat4 PlaneModel(1.0f);
	PlaneModel = translate(PlaneModel, glm::vec3(0.0f, -1.34f, 0.0f));
	PlaneModel = scale(PlaneModel, glm::vec3(5.0f));
	plane.setModelMatrix(PlaneModel);

	// create matrix uniforms and add them to the shader program
	const auto projUniform = std::make_shared<Uniform<glm::mat4>>("ProjectionMatrix", proj);
	auto viewUniform = std::make_shared<Uniform<glm::mat4>>("ViewMatrix", view);
	auto modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", model);
	auto cameraPosUniform = std::make_shared<Uniform<glm::vec3>>("cameraPos", cameraPos);

	sp.addUniform(projUniform);
	sp.addUniform(viewUniform);
	sp.addUniform(modelUniform);
	sp.addUniform(cameraPosUniform);

	skyboxSP.addUniform(projUniform);
	skyboxSP.addUniform(viewUniform);

	glm::vec3 ambient(0.5f);
	const auto ambientLightUniform = std::make_shared<Uniform<glm::vec3>>("lightAmbient", ambient);
	sp.addUniform(ambientLightUniform);

	// "generate" lights
	std::vector<LightInfo> lvec;
	for (int i = 0; i < 5; i++)
	{
		LightInfo li;
		const glm::mat4 rotMat = rotate(glm::mat4(1.0f), glm::radians(i * (360.0f / 5.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
		li.pos = rotMat * (glm::vec4(i * 3.0f, i * 3.0f, i * 3.0f, 1.0f) + glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
		li.col = normalize(glm::vec3((i) % 5, (i + 1) % 5, (i + 2) % 5));
		if (i % 2)
		{
			li.col = normalize(glm::vec3((i - 1) % 5, (i) % 5, (i + 1) % 5));
			li.col = normalize(glm::vec3(1.0f) - li.col);
		}
		std::cout << to_string(li.col) << std::endl;
		if (i == 3)
		{
			li.spot_cutoff = 0.1f;
		}
		else
		{
			li.spot_cutoff = 0.0f;
		}
		li.spot_direction = normalize(glm::vec3(0.0f) - glm::vec3(li.pos));
		li.spot_exponent = 1.0f;

		lvec.push_back(li);
	}

	// set up materials
	std::vector<MaterialInfo> mvec;
	MaterialInfo m;
	m.diffColor = glm::vec3(0.9f);
	m.kd = 0.5f;
	m.specColor = glm::vec3(0.9f);
	m.ks = 3.0f;
	m.shininess = 100.0f;
	m.kt = 0.0f;
	m.reflective = 0;
	mvec.push_back(m);

	MaterialInfo m2;
	m2.diffColor = glm::vec3(0.9f);
	m2.kd = 0.3f;
	m2.specColor = glm::vec3(0.9f);
	m2.ks = 0.3f;
	m2.shininess = 20.0f;
	m2.kt = 0.0f;
	m2.reflective = 1;
	mvec.push_back(m2);

	// first material is for bunny, second for plane
	bunny->setMaterialID(0);
	plane.setMaterialID(1);

	// create buffers for materials and lights
	Buffer lightBuffer(GL_SHADER_STORAGE_BUFFER);
	lightBuffer.setStorage(lvec, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
	lightBuffer.bindBase(0);

	Buffer materialBuffer(GL_SHADER_STORAGE_BUFFER);
	materialBuffer.setStorage(mvec, GL_DYNAMIC_STORAGE_BIT);
	materialBuffer.bindBase(1);

	FogInfo f;
	f.start = 0.0f;
	f.end = 10.0f;
	f.density = 0.1f;
	f.col = glm::vec3(0.1f);
	f.mode = 3;
	std::vector<FogInfo> fogvec{f};
	Buffer fogBuffer(GL_SHADER_STORAGE_BUFFER);
	fogBuffer.setStorage(fogvec, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
	fogBuffer.bindBase(2);

	const glm::vec4 clearColor(0.1f);

	// values for GUI-controllable uniforms
	bool flat = false;
	bool toon = false;
	int levels = 3;

	// shading uniforms
	auto flatUniform = std::make_shared<Uniform<bool>>("useFlat", flat);
	auto toonUniform = std::make_shared<Uniform<bool>>("useToon", toon);
	auto levelsUniform = std::make_shared<Uniform<int>>("levels", levels);
	auto MaterialIDUniform = std::make_shared<Uniform<int>>("matIndex", bunny->getMaterialID());

	sp.addUniform(flatUniform);
	sp.addUniform(toonUniform);
	sp.addUniform(levelsUniform);
	sp.addUniform(MaterialIDUniform);

	Timer timer;

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_DEPTH_TEST);

	std::vector<glm::vec3> rotations(5, glm::vec3(0.0f));

	const float deltaAngle = 0.1f;

	bool useFBO = true;
	bool renderSkyBox = true;

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		timer.start();
		const glm::mat4 newModel = rotate(bunny->getModelMatrix(), glm::radians(deltaAngle), glm::vec3(0.0f, 1.0f, 0.0f));
		bunny->setModelMatrix(newModel);

		glfwPollEvents();
		ImGui_ImplGlfwGL3_NewFrame();
		sp.showReloadShaderGUI(vs, fs, "Forward Lighting");
		fboSP.showReloadShaderGUI(fboVS, fboFS, "Postprocessing");
		{
			ImGui::SetNextWindowSize(ImVec2(100, 100), ImGuiSetCond_FirstUseEver);
			ImGui::Begin("Lighting settings");
			if (ImGui::Checkbox("Flat Shading", &flat)) flatUniform->setContent(flat);
			if (ImGui::Checkbox("Toon Shading", &toon)) toonUniform->setContent(toon);
			if (toon)
			{
				if (ImGui::SliderInt("Toon Shading Levels", &levels, 1, 5))
					levelsUniform->setContent(levels);
			}
			if (ImGui::SliderInt("Fog Mode", &fogvec.at(0).mode, 0, 3))
			{
				const auto fogModeOffset = sizeof(f.col) + sizeof(f.start) + sizeof(f.end) + sizeof(f.density);
				fogBuffer.setPartialContentMapped(fogvec.at(0).mode, fogModeOffset);
			}
			if (ImGui::SliderFloat3("Fog Color", value_ptr(fogvec.at(0).col), 0.0f, 1.0f))
			{
				fogBuffer.setPartialContentMapped(fogvec.at(0).col, 0);
			}
			if (ImGui::Button("Reset Fog Color"))
			{
				fogvec.at(0).col = glm::vec3(0.1f);
				fogBuffer.setPartialContentMapped(fogvec.at(0).col, 0);
			}
			ImGui::End();

			ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
			//ImGui::SetNextWindowPos(ImVec2(20, 150));
			ImGui::Begin("Lights settings");
			for (int i = 0; i < lvec.size(); ++i)
			{
				std::stringstream n;
				n << i;
				ImGui::Text((std::string("Light ") + n.str()).c_str());
				if (ImGui::SliderFloat3((std::string("Color ") + n.str()).c_str(), value_ptr(lvec.at(i).col), 0.0f, 1.0f))
				{
					const auto colOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, col);
					lightBuffer.setContentSubData(lvec.at(i).col, colOffset);
				}
				if (ImGui::SliderFloat((std::string("Cutoff ") + n.str()).c_str(), &lvec.at(i).spot_cutoff, 0.0f, 0.5f))
				{
					const auto spotCutoffOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_cutoff);
					lightBuffer.setContentSubData(lvec.at(i).spot_cutoff, spotCutoffOffset);
				}
				if (ImGui::SliderFloat((std::string("Exponent ") + n.str()).c_str(), &lvec.at(i).spot_exponent, 0.0f, 100.0f))
				{
					const auto spotCutoffExpOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_exponent);
					lightBuffer.setContentSubData(lvec.at(i).spot_exponent, spotCutoffExpOffset);
				}
				if (ImGui::SliderFloat3((std::string("Rotate ") + n.str()).c_str(), value_ptr(rotations.at(i)), 0.0f, 360.0f))
				{
					const auto posOffset = i * sizeof(lvec.at(i));
					const glm::mat4 rotx = rotate(glm::mat4(1.0f), glm::radians(rotations.at(i).x), glm::vec3(1.0f, 0.0f, 0.0f));
					const glm::mat4 rotxy = rotate(rotx, glm::radians(rotations.at(i).y), glm::vec3(0.0f, 1.0f, 0.0f));
					const glm::mat4 rotxyz = rotate(rotxy, glm::radians(rotations.at(i).z), glm::vec3(0.0f, 0.0f, 1.0f));
					const glm::vec3 newPos = rotxyz * lvec.at(i).pos;
					lightBuffer.setContentSubData(newPos, posOffset);
					lvec.at(i).spot_direction = normalize(glm::vec3(0.0f) - newPos);
					const auto spotDirOffset = i * sizeof(lvec.at(i)) + offsetof(LightInfo, spot_direction);
					lightBuffer.setContentSubData(lvec.at(i).spot_direction, spotDirOffset);
				}
				// maps memory to access it by GUI -- probably very bad performance-wise
				const auto positionOffset = i * sizeof(lvec.at(i));
				//lightBuffer.bind();
				float* ptr = lightBuffer.mapBufferContent<float>(sizeof(float) * 3, positionOffset, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
				ImGui::SliderFloat3((std::string("Position (conflicts rotation) ") + n.str()).c_str(), ptr, -30.0f, 30.0f);
				lightBuffer.unmapBuffer();
			}
			ImGui::End();

			ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
			ImGui::Begin("Rendering Settings");
			if (ImGui::Checkbox("Render to FBO", &useFBO));
			if (useFBO)
			{
				if (ImGui::Checkbox("Grayscale", &useGrayscale))
					grayscaleuniform->setContent(useGrayscale);
				if (ImGui::Checkbox("Blur", &useBlur))
				{
					blurUniform->setContent(useBlur);
					if (useBlur && useSharpen)
					{
						useSharpen = false;
						sharpenUniform->setContent(useSharpen);
					}
				}
				if (ImGui::Checkbox("Sharpen", &useSharpen))
				{
					sharpenUniform->setContent(useSharpen);
					if (useBlur && useSharpen)
					{
						useBlur = false;
						blurUniform->setContent(useBlur);
					}
				}
			}
			if (ImGui::Checkbox("Render Skybox", &renderSkyBox));
			ImGui::End();
		}

		//ImGui::ShowTestWindow();
		if (useFBO)
			fbo.bind(); // render into fbo

		glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		camera.update(window);

		viewUniform->setContent(camera.getView());
		cameraPosUniform->setContent(camera.getPosition());
		sp.use();

		// prepare first mesh (bunny)
		modelUniform->setContent(bunny->getModelMatrix());
		MaterialIDUniform->setContent(bunny->getMaterialID());
		sp.updateUniforms();

		bunny->draw();

		// prepare plane
		modelUniform->setContent(plane.getModelMatrix());
		MaterialIDUniform->setContent(plane.getMaterialID());
		sp.updateUniforms();

		plane.draw();

		if (renderSkyBox)
		{
			// render skybox last
			glDepthFunc(GL_LEQUAL);
			glDisable(GL_CULL_FACE);
			skyboxSP.use();
			cube.draw();
			glEnable(GL_CULL_FACE);
			glDepthFunc(GL_LEQUAL);
		}

		if (useFBO)
		{
			fbo.unbind(); // render to screen now
			glDisable(GL_DEPTH_TEST);
			glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			fboSP.use();
			fboQuad.draw();
			glEnable(GL_DEPTH_TEST);
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
