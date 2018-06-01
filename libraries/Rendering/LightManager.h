#pragma once

#include <glbinding/gl/gl.h>
using namespace gl;

#include <vector>
#include <memory>
#include "Mesh.h"
#include "Light.h"

class LightManager
{
public:

    LightManager();
    explicit LightManager(std::vector<std::shared_ptr<Light>> lights);

    void uploadLightsToGPU();

    bool showLightGUIs();

    bool showLightGUIsContent();

    void renderShadowMaps(const std::vector<std::shared_ptr<Mesh>>& meshes);
    void renderShadowMaps(const ModelImporter& mi);
    void renderShadowMapsCulled(const ModelImporter& scene);

    void updateLightParams();
    void updateLightParams(std::shared_ptr<Light> light);

    void addLight(std::shared_ptr<Light> light);

    std::vector<std::shared_ptr<Light>> getLights() const;

private:

    Buffer m_lightsBuffer{ GL_SHADER_STORAGE_BUFFER };
    std::vector<std::shared_ptr<Light>> m_lightList;
    int m_e = 0;

};