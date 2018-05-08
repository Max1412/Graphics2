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
    glm::mat4 lightSpaceMatrix;
    glm::vec3 color;                // all
    int type = -1;                  // 0 directional, 1 point light, 2 spot light
    glm::vec3 position;             // spot, point
    float constant = -1.0f;         // spot, point
    glm::vec3 direction;            // dir, spot
    float linear = -1.0f;         // spot, point
    float quadratic = -1.0f;      // spot, point
    float cutOff = -1.0f;         // spot
    float outerCutOff = -1.0f;    // spot
    int32_t pad; // TODO PAD
    int64_t shadowMap;
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

    // explicit Light(LightType type, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024));
    Light(glm::vec3 color, glm::vec3 direction, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024)); // DIRECTIONAL
    Light(glm::vec3 color, glm::vec3 position, float constant, float linear, float quadratic, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024)); // POINT
    Light(glm::vec3 color, glm::vec3 position, glm::vec3 direction, float constant, float linear, float quadratic, float cutOff, float outerCutOff, glm::ivec2 shadowMapRes = glm::ivec2(1024, 1024)); // SPOT

    void renderShadowMap(const std::vector<std::shared_ptr<Mesh>>& scene);

    const GPULight& getGpuLight() const;

    void recalculateLightSpaceMatrix();

    bool showLightGUI(const std::string& name = "Light");

    // getters & setters 
    void setColor(glm::vec3 col);
    glm::vec3 getColor() const;

    void setPosition(glm::vec3 pos);
    glm::vec3 getPosition() const;

    void setDirection(glm::vec3 dir);
    glm::vec3 getDirection() const;

    void setConstant(float constant);
    float getConstant() const;

    void setLinear(float linear);
    float getLinear() const;

    void setQuadratic(float quadratic);
    float getQuadratic() const;

    void setCutoff(float cutoff);
    float getCutoff() const;

    void setOuterCutoff(float cutOff);
    float getOuterCutoff() const;

private:
    void checkParameters();

    LightType m_type;

    bool m_hasShadowMap = true;
    glm::ivec2 m_shadowMapRes;

    glm::mat4 m_lightProjection;
    glm::mat4 m_lightView;

    Texture m_shadowTexture;
    FrameBuffer m_shadowMapFBO;
    ShaderProgram m_genShadowMapProgram;

    std::shared_ptr<Uniform<glm::mat4>> m_modelUniform;
    std::shared_ptr<Uniform<glm::mat4>> m_lightSpaceUniform;

    GPULight m_gpuLight;

    // Lighting parameters
    glm::vec3 m_color; // all
    glm::vec3 m_position; // spot, point
    glm::vec3 m_direction; // dir, spot
    float m_constant = -1.0f; // spot, point
    float m_linear = -1.0f;; // spot, point
    float m_quadratic = -1.0f;; // spot, point
    float m_cutOff = -1.0f;; // spot
    float m_outerCutOff = -1.0f;; // spot

};

class LightManager
{
public:

    LightManager();
    explicit LightManager(std::vector<std::shared_ptr<Light>> lights);

    void uploadLightsToGPU();

    bool showLightGUIs();

    void renderShadowMaps(const std::vector<std::shared_ptr<Mesh>>& scene);
    void updateLightParams();
    void updateLightParams(std::shared_ptr<Light> light);

    void addLight(std::shared_ptr<Light> light);

    std::vector<std::shared_ptr<Light>> getLights() const;

private:

    Buffer m_lightsBuffer{ GL_SHADER_STORAGE_BUFFER };
    std::vector<std::shared_ptr<Light>> m_lightList;

};