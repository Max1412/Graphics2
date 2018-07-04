#include "LightManager.h"
#include "imgui/imgui.h"
#include <execution>

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

void LightManager::bindLightBuffer() const
{
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

bool LightManager::showLightGUIsContent()
{
    bool changed = false;
    int index = 0;
    std::for_each(m_lightList.begin(), m_lightList.end(), [this, &changed, &index](auto& light)
    {
        index++;
        ImGui::Separator();
        if (light->showLightGUIContent(std::string("Light ") + std::to_string(index)))
        {
            updateLightParams(light);
            changed = true;
        }
    });
    return changed;
}

void LightManager::renderShadowMaps(const std::vector<std::shared_ptr<Mesh>>& meshes)
{
    std::for_each(m_lightList.begin(), m_lightList.end(), [&meshes](auto& light)
    {
        light->renderShadowMap(meshes);
    });
}

void LightManager::renderShadowMaps(const ModelImporter& mi)
{
    std::for_each(m_lightList.begin(), m_lightList.end(), [&mi](auto& light)
    {
        light->renderShadowMap(mi);
    });
}

void LightManager::renderShadowMapsCulled(const ModelImporter& scene)
{
    std::for_each(m_lightList.begin(), m_lightList.end(), [&scene](auto& light)
    {
        light->renderShadowMapCulled(scene);
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

void LightManager::setOuterSceneBoundingBoxToAllLights(const glm::mat2x4& outerSceneBoundingBox)
{
    std::for_each(std::execution::par, m_lightList.begin(), m_lightList.end(),
        [&outerSceneBoundingBox](auto& light) { light->setOuterBoundingBox(outerSceneBoundingBox); });
}