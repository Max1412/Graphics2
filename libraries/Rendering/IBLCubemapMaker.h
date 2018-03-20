#pragma once
#include <filesystem>
#include "Cubemap.h"
#include "SkyBoxCube.h"
#include "Uniform.h"
#include "ShaderProgram.h"

class IBLCubemapMaker
{
public:
    IBLCubemapMaker(const std::experimental::filesystem::path& filename);
    Cubemap getEnvironmentCubemap() const;
    Cubemap getIrradianceCubemap() const;
    Cubemap getSpecularCubemap() const;
    Texture getBRDFLUT() const;

    void draw(glm::mat4 view, glm::mat4 proj);


private:
    Cubemap m_targetCubemap;
    Cubemap m_irradianceCubemap;
    Cubemap m_specularCubemap;

    Buffer m_iblSkyboxTextureBuffer;
    Buffer m_irrCalcTextureBuffer;

    Shader m_skyBoxVS;
    Shader m_skyBoxFS;

    ShaderProgram m_iblSkyboxSP;

    SkyBoxCube m_cube;
    std::shared_ptr<Uniform<glm::mat4>> m_projUniform;
    std::shared_ptr<Uniform<glm::mat4>> m_viewUniform;

    Texture m_brdfLUT;
};
