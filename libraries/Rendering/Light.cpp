#include "Light.h"
#include <glm/gtc/matrix_transform.inl>
#include "imgui/imgui.h"
#include <glm/gtc/type_ptr.hpp>
#include <sstream>

using namespace gl;

Light::Light(glm::vec3 color, glm::vec3 direction, glm::ivec2 shadowMapRes) // DIRECTIONAL
: m_type(LightType::directional), m_shadowMapRes(shadowMapRes),
m_shadowTexture(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST), m_shadowMapFBO(GL_DEPTH_ATTACHMENT, m_shadowTexture), m_genShadowMapProgram("lightTransform.vert", "nothing.frag", BufferBindings::g_definitions),
m_color(color), m_direction(direction)
{
    checkParameters();

    // init shadowMap
    m_shadowTexture.initWithoutData(m_shadowMapRes.x, m_shadowMapRes.y, GL_DEPTH_COMPONENT32F);
    m_shadowTexture.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

    m_modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", glm::mat4(1.0f));
    m_lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", glm::mat4(1.0f));

    m_genShadowMapProgram.addUniform(m_modelUniform);
    m_genShadowMapProgram.addUniform(m_lightSpaceUniform);

    recalculateLightSpaceMatrix();

    // init gpu struct
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = m_color;
    m_gpuLight.direction = m_direction;
    m_gpuLight.shadowMap = m_shadowTexture.generateHandle();
}

Light::Light(glm::vec3 color, glm::vec3 position, float constant, float linear, float quadratic, glm::ivec2 shadowMapRes) // POINT
: m_type(LightType::point), m_shadowMapRes(shadowMapRes),
m_shadowTexture(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST), m_shadowMapFBO(GL_DEPTH_ATTACHMENT, m_shadowTexture), m_genShadowMapProgram("lightTransform.vert", "nothing.frag", BufferBindings::g_definitions),
m_color(color), m_position(position), m_constant(constant), m_linear(linear), m_quadratic(quadratic)
{
    checkParameters();

    // init gpu struct
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = m_color;
    m_gpuLight.position = m_position;
    m_gpuLight.constant = m_constant;
    m_gpuLight.linear = m_linear;
    m_gpuLight.quadratic = m_quadratic;

    throw std::runtime_error("POINT LIGHTS NOT SUPPORTED YET");
}

Light::Light(glm::vec3 color, glm::vec3 position, glm::vec3 direction, float constant, float linear, float quadratic, float cutOff, float outerCutOff, glm::ivec2 shadowMapRes) // SPOT
    : m_type(LightType::spot), m_shadowMapRes(shadowMapRes),
    m_shadowTexture(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST), m_shadowMapFBO(GL_DEPTH_ATTACHMENT, m_shadowTexture), m_genShadowMapProgram("lightTransform.vert", "nothing.frag", BufferBindings::g_definitions),
    m_color(color), m_position(position), m_direction(direction), m_constant(constant), m_linear(linear), m_quadratic(quadratic), m_cutOff(cutOff), m_outerCutOff(outerCutOff)
{
    checkParameters();

    // init shadowMap
    m_shadowTexture.initWithoutData(m_shadowMapRes.x, m_shadowMapRes.y, GL_DEPTH_COMPONENT32F);
    m_shadowTexture.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

    m_modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", glm::mat4(1.0f));
    m_lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", glm::mat4(1.0f));

    m_genShadowMapProgram.addUniform(m_modelUniform);
    m_genShadowMapProgram.addUniform(m_lightSpaceUniform);

    recalculateLightSpaceMatrix();

    // init gpu struct
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = m_color;
    m_gpuLight.position = m_position;
    m_gpuLight.direction = m_direction;
    m_gpuLight.constant = m_constant;
    m_gpuLight.linear = m_linear;
    m_gpuLight.quadratic = m_quadratic;
    m_gpuLight.cutOff = m_cutOff;
    m_gpuLight.outerCutOff = m_outerCutOff;
    m_gpuLight.shadowMap = m_shadowTexture.generateHandle();
}


void Light::renderShadowMap(const std::vector<std::shared_ptr<Mesh>>& scene)
{
    if (!m_hasShadowMap)
        return;

    recalculateLightSpaceMatrix();

    //store old viewport
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    //set sm settings
    m_genShadowMapProgram.use();
    glViewport(0, 0, m_shadowMapRes.x, m_shadowMapRes.y);
    m_shadowMapFBO.bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glCullFace(GL_FRONT);

    //render scene
    std::for_each(scene.begin(), scene.end(), [&](auto& mesh)
    {
        m_modelUniform->setContent(mesh->getModelMatrix());
        m_genShadowMapProgram.updateUniforms();
        mesh->draw();
    });

    //restore previous rendering settings
    m_shadowMapFBO.unbind();
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glCullFace(GL_BACK);
}

void Light::recalculateLightSpaceMatrix()
{
    if (m_type == LightType::directional) 
    {
        const float nearPlane = 3.0f, farPlane = 18.0f;
        m_lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, nearPlane, farPlane);

        m_lightView = lookAt(m_gpuLight.position,
            glm::vec3(0.0f), // aimed at the center
            glm::vec3(0.0f, 1.0f, 0.0f));
    }
    else if (m_type == LightType::spot) 
    {
        const float nearPlane = 3.0f, farPlane = 18.0f;
        // NOTE: ACOS BECAUSE CUTOFF HAS COS BAKED IN
        m_lightProjection = glm::perspective(2.0f*glm::acos(m_gpuLight.cutOff), static_cast<float>(m_shadowMapRes.x) / static_cast<float>(m_shadowMapRes.y), nearPlane, farPlane);
        // TODO WHAT TO DO WITH OUTER CUTOFF???
        m_lightView = lookAt(m_gpuLight.position,
            m_gpuLight.position + m_gpuLight.direction, // aimed at the center
            glm::vec3(0.0f, 1.0f, 0.0f));
    }

    m_gpuLight.lightSpaceMatrix = m_lightProjection * m_lightView;
    m_lightSpaceUniform->setContent(m_gpuLight.lightSpaceMatrix);
}

void Light::checkParameters()
{
    // TODO implement this if there are constructors that do not initalize all the params for the specific type
}

const GPULight& Light::getGpuLight() const
{
    return m_gpuLight;
}

void Light::setPosition(glm::vec3 pos)
{
    m_position = pos;
    m_gpuLight.position = pos;
    recalculateLightSpaceMatrix();
}

void Light::setColor(glm::vec3 col)
{
    m_color = col;
    m_gpuLight.color = col;
}

void Light::setCutoff(float cutoff)
{
    m_cutOff = cutoff;
    m_gpuLight.cutOff = cutoff;
    recalculateLightSpaceMatrix();
}

void Light::setDirection(glm::vec3 dir)
{
    m_direction = dir;
    m_gpuLight.direction = dir;
    recalculateLightSpaceMatrix();
}


void Light::setConstant(float constant)
{
    m_constant = constant;
    m_gpuLight.constant = constant;
}

float Light::getConstant() const
{
    return m_gpuLight.constant;
}

void Light::setLinear(float linear)
{
    m_linear = linear;
    m_gpuLight.linear = linear;
}

float Light::getLinear() const
{
    return m_gpuLight.linear;
}

void Light::setQuadratic(float quadratic)
{
    m_quadratic = quadratic;
    m_gpuLight.quadratic = quadratic;
}

float Light::getQuadratic() const
{
    return m_gpuLight.quadratic;
}

glm::vec3 Light::getPosition() const
{
    return m_gpuLight.position;
}

glm::vec3 Light::getColor() const
{
    return m_gpuLight.color;
}

float Light::getCutoff() const
{
    return m_gpuLight.cutOff;
}

void Light::setOuterCutoff(float cutOff)
{
    m_outerCutOff = cutOff;
    m_gpuLight.outerCutOff = cutOff;
}

float Light::getOuterCutoff() const
{
    return m_gpuLight.outerCutOff;
}

glm::vec3 Light::getDirection() const
{
    return m_gpuLight.direction;
}


bool Light::showLightGUI(const std::string& name)
{
    ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
    std::stringstream fullName;
    fullName << name << " (Type: " << static_cast<int>(m_type) << ")";
    ImGui::Begin(fullName.str().c_str());
    bool lightChanged = false;
    if (ImGui::SliderFloat3((std::string("Color ") + name).c_str(), value_ptr(m_gpuLight.color), 0.0f, 1.0f))
    {
        m_color = m_gpuLight.color;
        lightChanged = true;
    }
    if(m_type == LightType::directional || m_type == LightType::spot)
    {
        if (ImGui::SliderFloat3((std::string("Direction ") + name).c_str(), value_ptr(m_gpuLight.direction), -1.0f, 1.0f))
        {
            m_direction = m_gpuLight.direction;
            lightChanged = true;
        }
    }
    if (m_type == LightType::spot)
    {
        if (ImGui::SliderFloat((std::string("Cutoff ") + name).c_str(), &m_gpuLight.cutOff, 0.0f, glm::radians(90.0f)))
        {
            m_cutOff = m_gpuLight.cutOff;
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Outer cutoff ") + name).c_str(), &m_gpuLight.outerCutOff, 0.0f, 10.0f))
        {
            m_outerCutOff = m_gpuLight.outerCutOff;
            lightChanged = true;
        }
    }
    if (m_type == LightType::spot || m_type == LightType::point)
    {
        if (ImGui::SliderFloat3((std::string("Position ") + name).c_str(), value_ptr(m_gpuLight.position), -10.0f, 10.0f))
        {
            m_position = m_gpuLight.position;
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Constant ") + name).c_str(), &m_gpuLight.constant, 0.0f, 10.0f))
        {
            m_constant = m_gpuLight.constant;
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Linear ") + name).c_str(), &m_gpuLight.linear, 0.0f, 10.0f))
        {
            m_linear = m_gpuLight.linear;
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Quadratic ") + name).c_str(), &m_gpuLight.quadratic, 0.0f, 10.0f))
        {
            m_quadratic = m_gpuLight.quadratic;
            lightChanged = true;
        }
    }
    ImGui::End();

    if (lightChanged) recalculateLightSpaceMatrix();
    // TODO only recalculate if necessary

    return lightChanged;
}

// Light Manager
LightManager::LightManager()
{
}

LightManager::LightManager(std::vector<std::shared_ptr<Light>> lights)
{
    m_lightList = lights;
}

void LightManager::uploadLightsToGPU()
{
    std::vector<GPULight> gpuLights;
    std::for_each(m_lightList.begin(), m_lightList.end(), [&gpuLights](auto& light)
    {
        light->recalculateLightSpaceMatrix();
        gpuLights.push_back(light->getGpuLight());
    });

    m_lightsBuffer.setStorage(gpuLights, GL_DYNAMIC_STORAGE_BIT);
    m_lightsBuffer.bindBase(BufferBindings::Binding::lights);
}

bool LightManager::showLightGUIs()
{
    bool changed = false;
    int index = 0;
    std::for_each(m_lightList.begin(), m_lightList.end(), [this, &changed, &index](auto& light)
    {
        index++;
        if (light->showLightGUI(std::string("Light ") + std::to_string(index)))
        {
            updateLightParams(light);
            changed = true;
        }
    });
    return changed;
}

void LightManager::renderShadowMaps(const std::vector<std::shared_ptr<Mesh>>& scene)
{
    std::for_each(m_lightList.begin(), m_lightList.end(), [&scene](auto& light)
    {
        light->renderShadowMap(scene);
    });
}

void LightManager::updateLightParams()
{
    std::vector<GPULight> gpuLights;
    std::for_each(m_lightList.begin(), m_lightList.end(), [&gpuLights](auto& light)
    {
        light->recalculateLightSpaceMatrix();
        gpuLights.push_back(light->getGpuLight());
    });

    m_lightsBuffer.setContentSubData(gpuLights, 0);
    m_lightsBuffer.bindBase(BufferBindings::Binding::lights);
}

void LightManager::updateLightParams(std::shared_ptr<Light> light)
{
    size_t index = std::distance(m_lightList.begin(), std::find(m_lightList.begin(), m_lightList.end(), light));
    if (index < m_lightList.size())
    {
        m_lightList[index]->recalculateLightSpaceMatrix();
        m_lightsBuffer.setContentSubData(m_lightList[index]->getGpuLight(), index * sizeof(GPULight));
        m_lightsBuffer.bindBase(BufferBindings::Binding::lights);
    }
    else
    {
        throw std::runtime_error("Tried to update a light that was not added to this LightManager!");
    }
}

void LightManager::addLight(std::shared_ptr<Light> light)
{
    m_lightList.push_back(light);
}

std::vector<std::shared_ptr<Light>> LightManager::getLights() const
{
    return m_lightList;
}
