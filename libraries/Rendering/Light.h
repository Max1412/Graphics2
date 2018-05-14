#pragma once

#include <glbinding/gl/gl.h>
#include <glm/glm.hpp>
#include "Buffer.h"
#include "Texture.h"
#include "ShaderProgram.h"
#include "Mesh.h"
#include "FrameBuffer.h"


using namespace gl;

struct GPULight
{
    glm::vec3 position; 
    int type; //0 directional, 1 point light, 2 spot light
    glm::vec3 color;
    float spotCutoff;
    glm::vec3 spotDirection;
    float spotExponent;
    glm::mat4 lightSpaceMatrix;
    uint64_t shadowMap; //can be sampler2D or samplerCube
    float pad1, pad2;
};

enum class LightType : int
{
    directional = 0,
    point = 1,
    spot = 2
};

class Light
{
public:

    Light(LightType type, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024));
    Light(LightType type, glm::vec3 position, glm::ivec2 shadowMapRes = glm::ivec2(1024,1024));
    Light(LightType type, glm::vec3 position, glm::vec3 color, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024));
    Light(LightType type, glm::vec3 position, glm::vec3 color, glm::vec3 spotDir, float spotCutoff, float spotExponent, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024));

    void renderShadowMap(const std::vector<std::shared_ptr<Mesh>>& scene);

    const GPULight& getGpuLight() const;

    void recalculateLightSpaceMatrix();

    bool showLightGUI(std::string name = "Light");
	bool showLightGUIContent(std::string name = "Light");

    void setPosition(glm::vec3 pos);
    void setColor(glm::vec3 col);
    void setSpotCutoff(float cutoff);
    void setSpotExponent(float exp);
    void setSpotDirection(glm::vec3 dir);

    glm::vec3 getPosition() const;
    glm::vec3 getColor() const;
    float getSpotCutoff() const;
    float getSpotExponent() const;
    glm::vec3 getSpotDirection() const;

private:

    GPULight m_gpuLight;

    LightType m_type;
    bool m_hasShadowMap = false;
    glm::ivec2 m_shadowMapRes;

    glm::mat4 m_lightProjection;
    glm::mat4 m_lightView;

    std::shared_ptr<Texture> m_shadowTexture;
    std::unique_ptr<FrameBuffer> m_shadowMapFBO;
    std::unique_ptr<ShaderProgram> m_genShadowMapProgram;
    std::shared_ptr<Uniform<glm::mat4>> m_modelUniform;
    std::shared_ptr<Uniform<glm::mat4>> m_lightSpaceUniform;

    void init(glm::vec3 position, glm::vec3 color, glm::vec3 spotDir, float spotCutoff, float spotExponent);
};

class LightManager
{
public:

    LightManager();
    explicit LightManager(std::vector<std::shared_ptr<Light>> lights);

    void uploadLightsToGPU();

    bool showLightGUIs();
	bool showLightGUIsContent();

    void renderShadowMaps(const std::vector<std::shared_ptr<Mesh>>& scene);
    void updateLightParams();
    void updateLightParams(std::shared_ptr<Light> light);

    void addLight(std::shared_ptr<Light> light);

    std::vector<std::shared_ptr<Light>> getLights() const;

private:

    Buffer m_lightsBuffer{ GL_SHADER_STORAGE_BUFFER };
    std::vector<std::shared_ptr<Light>> m_lightList;

};