#pragma once

#include <glbinding/gl/gl.h>
#include <glm/glm.hpp>
#include "Buffer.h"
#include "Texture.h"
#include "ShaderProgram.h"
#include "Mesh.h"
#include "FrameBuffer.h"


class ModelImporter;
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
    int64_t shadowMap;
    float quadratic = -1.0f;      // spot, point
    float cutOff = -1.0f;         // spot
    float outerCutOff = -1.0f;    // spot
    int32_t pad, pad2, pad3; // TODO PAD
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

    void renderShadowMap(const ModelImporter& mi);
    void renderShadowMapCulled(const ModelImporter& mi);

    const GPULight& getGpuLight() const;

    void recalculateLightSpaceMatrix();

    bool showLightGUI(const std::string& name = "Light");
	bool showLightGUIContent(const std::string& name = "Light");

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

    LightType getType() const;

private:
    void checkParameters();

    LightType m_type;

    bool m_hasShadowMap = true;
    glm::ivec2 m_shadowMapRes;

    glm::mat4 m_lightProjection;
    glm::mat4 m_lightView;

    std::shared_ptr<Texture> m_shadowTexture;
    FrameBuffer m_shadowMapFBO;
    ShaderProgram m_genShadowMapProgram;

    std::shared_ptr<Uniform<glm::mat4>> m_modelUniform;
    std::shared_ptr<Uniform<glm::mat4>> m_lightSpaceUniform;
    std::shared_ptr<Uniform<glm::vec3>> m_lightPosUniform;


    GPULight m_gpuLight;
};