#include "IBLCubemapMaker.h"
#include "Texture.h"
#include "SkyBoxCube.h"
#include "Shader.h"
#include "ShaderProgram.h"
#include "Cubemap.h"
#include <glm/gtc/matrix_transform.inl>
#include "FrameBuffer.h"
#include "Quad.h"

IBLCubemapMaker::IBLCubemapMaker(const std::experimental::filesystem::path& filename)
	: m_iblSkyboxTextureBuffer(GL_SHADER_STORAGE_BUFFER),
	  m_irrCalcTextureBuffer(GL_SHADER_STORAGE_BUFFER),
	  m_skyBoxVS("drawIBLskybox.vert", GL_VERTEX_SHADER),
	  m_skyBoxFS("drawIBLskybox.frag", GL_FRAGMENT_SHADER),
	  m_iblSkyboxSP(m_skyBoxVS, m_skyBoxFS)
{
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	/////////////////////////////////////////////
	// Step 1: HDR rectangular image -> cubemap (for display & further processing)
	// Skip this step when rendering a scene to a cubemap
	/////////////////////////////////////////////
	Texture iblTexture;
	iblTexture.loadFromFile(filename, GL_RGB16F, GL_RGB, GL_FLOAT, 0);
	iblTexture.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	iblTexture.generateHandle();

	Buffer iblTextureBuffer(GL_SHADER_STORAGE_BUFFER);
	iblTextureBuffer.setStorage(std::array<GLuint64, 1>{iblTexture.getHandle()}, GL_DYNAMIC_STORAGE_BIT);
	iblTextureBuffer.bindBase(5);

	const Shader IBLtoCubeVS("IBLtoCube.vert", GL_VERTEX_SHADER);
	const Shader IBLtoCubeFS("IBLtoCube.frag", GL_FRAGMENT_SHADER);
	ShaderProgram IBLtoCubeSP(IBLtoCubeVS, IBLtoCubeFS);
	IBLtoCubeSP.use();

	m_targetCubemap.initWithoutData(512, 512, GL_RGB16F, GL_RGB, GL_FLOAT);
	m_targetCubemap.generateHandle();

	GLuint captureFBO, captureRBO;
	glGenFramebuffers(1, &captureFBO);
	glGenRenderbuffers(1, &captureRBO);

	glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
	glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);

	glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
	std::array<glm::mat4, 6> captureViews =
	{
		lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
		lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))
	};

	m_projUniform = std::make_shared<Uniform<glm::mat4>>("projection", captureProjection);
	IBLtoCubeSP.addUniform(m_projUniform);

	m_viewUniform = std::make_shared<Uniform<glm::mat4>>("view", glm::mat4(1.0f));
	IBLtoCubeSP.addUniform(m_viewUniform);

	glViewport(0, 0, 512, 512);
	glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
	for (unsigned int i = 0; i < 6; ++i)
	{
		m_viewUniform->setContent(captureViews.at(i));
		IBLtoCubeSP.updateUniforms();
		// TODO maybe get this to work with DSA somehow?
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_targetCubemap.getName(), 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		m_cube.draw();
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/////// Intermediate Step
	// Setup to draw the cubemap in draw()
	///////
	m_iblSkyboxTextureBuffer.setStorage(std::array<GLuint64, 1>{m_targetCubemap.getHandle()}, GL_DYNAMIC_STORAGE_BIT);
	m_iblSkyboxTextureBuffer.bindBase(6);

	m_iblSkyboxSP.addUniform(m_projUniform);
	m_iblSkyboxSP.addUniform(m_viewUniform);

	/////////////////////////////////////////////
	// Step 2: Cubemap -> Convoluted irradiance map (for diffuse lighting)
	/////////////////////////////////////////////
	const Shader irradianceFS("irradianceMapCalc.frag", GL_FRAGMENT_SHADER);
	ShaderProgram irradiancePS(IBLtoCubeVS, irradianceFS); // reuse cubemap vertexshader
	irradiancePS.use();

	m_irradianceCubemap.initWithoutData(32, 32, GL_RGB16F, GL_RGB, GL_FLOAT);
	m_irradianceCubemap.generateHandle();

	glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
	glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

	irradiancePS.addUniform(m_projUniform);
	irradiancePS.addUniform(m_viewUniform);
	irradiancePS.updateUniforms();
	glViewport(0, 0, 32, 32);
	glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
	for (unsigned int i = 0; i < 6; ++i)
	{
		m_viewUniform->setContent(captureViews.at(i));
		irradiancePS.updateUniforms();
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_irradianceCubemap.getName(), 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		m_cube.draw();
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// uncomment this to render the blurry irradiance map instead of the environment map
	//m_iblSkyboxTextureBuffer.setContentSubData(m_irradianceCubemap.getHandle(), 0);

	/////////////////////////////////////////////
	// Step 3: Cubemap -> Prefiltered Specular Cubemap (for reflections/specular lighting)
	/////////////////////////////////////////////

	// generate new cubemap with mipmaps
	const unsigned int maxMipLevels = 5;
	m_specularCubemap.initWithoutData(128, 128, GL_RGB16F, GL_RGB, GL_FLOAT, maxMipLevels);
	m_specularCubemap.setMinMagFilter(GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR); // enable trilinear filtering
	m_specularCubemap.generateMipmap();
	m_specularCubemap.generateHandle();

	const Shader specularFS("prefilterEnvMap.frag", GL_FRAGMENT_SHADER);
	ShaderProgram specularSP(IBLtoCubeVS, specularFS); // reuse cubemap vertexshader
	specularSP.use();
	specularSP.addUniform(m_projUniform);
	specularSP.addUniform(m_viewUniform);

	float roughness = 0.0f;
	auto roughnessUniform = std::make_shared<Uniform<float>>("roughness", roughness);
	specularSP.addUniform(roughnessUniform);

	glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
	for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
	{
		// reisze framebuffer according to mip-level size.
		unsigned int mipWidth = 128 * pow(0.5, mip);
		unsigned int mipHeight = 128 * pow(0.5, mip);
		glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
		glViewport(0, 0, mipWidth, mipHeight);

		roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
		roughnessUniform->setContent(roughness);
		for (unsigned int i = 0; i < 6; ++i)
		{
			m_viewUniform->setContent(captureViews.at(i));
			specularSP.updateUniforms();
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			                       GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_specularCubemap.getName(), mip);

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			m_cube.draw();
		}
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//m_iblSkyboxTextureBuffer.setContentSubData(m_specularCubemap.getHandle(), 0);

	/////////////////////////////////////////////
	// Step 4: Generate BRDF LUT
	/////////////////////////////////////////////

	m_brdfLUT.initWithoutData(512, 512, GL_RG16F);
	m_brdfLUT.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	m_brdfLUT.generateHandle();

	glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
	glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_brdfLUT.getName(), 0);

	ShaderProgram brdfLutSP("texSFQ.vert", "brdfLUT.frag");
	Quad quad;
	glViewport(0, 0, 512, 512);
	brdfLutSP.use();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	quad.draw();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Cubemap IBLCubemapMaker::getEnvironmentCubemap() const
{
	return m_targetCubemap;
}

Cubemap IBLCubemapMaker::getIrradianceCubemap() const
{
	return m_irradianceCubemap;
}

Cubemap IBLCubemapMaker::getSpecularCubemap() const
{
	return m_specularCubemap;
}

Texture IBLCubemapMaker::getBRDFLUT() const
{
	return m_brdfLUT;
}

void IBLCubemapMaker::draw(glm::mat4 view, glm::mat4 proj)
{
	m_iblSkyboxSP.showReloadShaderGUI(m_skyBoxVS, m_skyBoxFS, "Skybox rendering");
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	m_iblSkyboxSP.use();
	m_viewUniform->setContent(view);
	m_projUniform->setContent(proj);
	m_iblSkyboxSP.updateUniforms();
	m_cube.draw();
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
}
