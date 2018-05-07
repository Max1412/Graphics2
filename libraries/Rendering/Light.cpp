#include "Light.h"
#include <glm/gtc/matrix_transform.inl>
#include "imgui/imgui.h"
#include <glm/gtc/type_ptr.hpp>

using namespace gl;

Light::Light(LightType type, glm::ivec2 shadowMapRes) : m_type(type), m_shadowMapRes(shadowMapRes)
{
    init(glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(0.0f), 0.0f, 0.0f);
}

Light::Light(LightType type, glm::vec3 position, glm::ivec2 shadowMapRes) : m_type(type), m_shadowMapRes(shadowMapRes)
{
    if (type == LightType::spot)
        std::cout << "WARNING: initialized spot light without valid spot parameters \n";
    init(position, glm::vec3(1.0f), glm::vec3(0.0f), 0.0f, 0.0f);
}

Light::Light(LightType type, glm::vec3 position, glm::vec3 color, glm::ivec2 shadowMapRes) : m_type(type), m_shadowMapRes(shadowMapRes)
{
    if (type == LightType::spot)
        std::cout << "WARNING: initialized spot light without valid spot parameters \n";
    init(position, color, glm::vec3(0.0f), 0.0f, 0.0f);
}

Light::Light(LightType type, glm::vec3 position, glm::vec3 color, glm::vec3 spotDir, float spotCutoff, float spotExponent, glm::ivec2 shadowMapRes) : 
    m_type(type), m_shadowMapRes(shadowMapRes)
{
    init(position, color, spotDir, spotCutoff, spotExponent);
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
    m_genShadowMapProgram->use();
    glViewport(0, 0, m_shadowMapRes.x, m_shadowMapRes.y);
    m_shadowMapFBO->bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glCullFace(GL_FRONT);

    //render scene
    std::for_each(scene.begin(), scene.end(), [&](auto& mesh)
    {
        m_modelUniform->setContent(mesh->getModelMatrix());
        m_genShadowMapProgram->updateUniforms();
        mesh->draw();
    });

    //restore previous rendering settings
    m_shadowMapFBO->unbind();
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glCullFace(GL_BACK);
}

const GPULight& Light::getGpuLight() const
{
    return m_gpuLight;
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
        m_lightProjection = glm::perspective(2.0f*m_gpuLight.spotCutoff, static_cast<float>(m_shadowMapRes.x) / static_cast<float>(m_shadowMapRes.y), nearPlane, farPlane);

        m_lightView = lookAt(m_gpuLight.position,
            m_gpuLight.position + m_gpuLight.spotDirection, // aimed at the center
            glm::vec3(0.0f, 1.0f, 0.0f));
    }

    m_gpuLight.lightSpaceMatrix = m_lightProjection * m_lightView;
    m_lightSpaceUniform->setContent(m_gpuLight.lightSpaceMatrix);
}

void Light::init(glm::vec3 position, glm::vec3 color, glm::vec3 spotDir, float spotCutoff, float spotExponent)
{
    m_gpuLight.position = position;
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = color;
    m_gpuLight.spotCutoff = spotCutoff;
    m_gpuLight.spotDirection = spotDir;
    m_gpuLight.spotExponent = spotExponent;

    if (m_shadowMapRes.x > 0 && m_shadowMapRes.y > 0)
    {
        if (m_type == LightType::point)
            std::cout << "WARNING: shadow mapping is currently not supported for point lights \n";

        m_shadowTexture = std::make_shared<Texture>(GL_TEXTURE_2D, GL_NEAREST, GL_NEAREST);
        m_shadowTexture->initWithoutData(m_shadowMapRes.x, m_shadowMapRes.y, GL_DEPTH_COMPONENT32F);
        m_shadowTexture->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_gpuLight.shadowMap = m_shadowTexture->generateHandle();

        m_shadowMapFBO = std::make_unique<FrameBuffer>(GL_DEPTH_ATTACHMENT, *m_shadowTexture);

        m_genShadowMapProgram = std::make_unique<ShaderProgram>("lightTransform.vert", "nothing.frag", BufferBindings::g_definitions);
        m_modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", glm::mat4(1.0f));
        m_lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", glm::mat4(1.0f));
        m_genShadowMapProgram->addUniform(m_modelUniform);
        m_genShadowMapProgram->addUniform(m_lightSpaceUniform);

        recalculateLightSpaceMatrix();

        m_hasShadowMap = true;
    }
}

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

void Light::setPosition(glm::vec3 pos)
{
    m_gpuLight.position = pos;
    recalculateLightSpaceMatrix();
}

void Light::setColor(glm::vec3 col)
{
    m_gpuLight.color = col;
}

void Light::setSpotCutoff(float cutoff)
{
    m_gpuLight.spotCutoff = cutoff;
    recalculateLightSpaceMatrix();
}

void Light::setSpotExponent(float exp)
{
    m_gpuLight.spotExponent = exp;
}

void Light::setSpotDirection(glm::vec3 dir)
{
    m_gpuLight.spotDirection = dir;
    recalculateLightSpaceMatrix();
}

glm::vec3 Light::getPosition() const
{
    return m_gpuLight.position;
}

glm::vec3 Light::getColor() const
{
    return m_gpuLight.color;
}

float Light::getSpotCutoff() const
{
    return m_gpuLight.spotCutoff;
}

float Light::getSpotExponent() const
{
    return m_gpuLight.spotExponent;
}

glm::vec3 Light::getSpotDirection() const
{
    return m_gpuLight.spotDirection;
}

bool Light::showLightGUI(std::string name)
{
    ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
    std::stringstream fullName;
    fullName << name << " (Type: " << static_cast<int>(m_type) << ")";
    ImGui::Begin(fullName.str().c_str());
    bool lightChanged = false;
    if (ImGui::SliderFloat3((std::string("Color ") + name).c_str(), value_ptr(m_gpuLight.color), 0.0f, 1.0f))
    {
        lightChanged = true;
    }
    if (ImGui::SliderFloat3((std::string("Position ") + name).c_str(), value_ptr(m_gpuLight.position), -10.0f, 10.0f))
    {
        lightChanged = true;
    }
    if (m_type == LightType::spot)
    {
        if (ImGui::SliderFloat((std::string("Cutoff ") + name).c_str(), &m_gpuLight.spotCutoff, 0.0f, glm::radians(90.0f)))
        {
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Exponent ") + name).c_str(), &m_gpuLight.spotExponent, 0.0f, 10.0f))
        {
            lightChanged = true;
        }
        if (ImGui::SliderFloat3((std::string("Drection ") + name).c_str(), value_ptr(m_gpuLight.spotDirection), -1.0f, 1.0f))
        {
            lightChanged = true;
        }
    }

    ImGui::End();

    if (lightChanged) recalculateLightSpaceMatrix();

    return lightChanged;
}
