#include "SimplexNoise.h"
#include "imgui/imgui.h"
#include <glm/detail/func_exponential.inl>

SimplexNoise::SimplexNoise(float scale, float speed, float densityFactor, float densityHeight) :
	m_noiseScale(scale), m_noiseSpeed(speed), m_densityFactor(densityFactor), m_densityHeight(densityHeight)
{
    m_time = static_cast<float>(glfwGetTime());

    {
		std::array<char, 256 * 256 * 4> pixels;
        for (int i = 0; i<256; i++)
            for (int j = 0; j<256; j++) 
            {
                int offset = (i * 256 + j) * 4;
                char value = m_perm[(j + m_perm[i]) & 0xFF];
                pixels[offset] = m_gradient4[2 * (value & 0x0F)][0] * 64 + 64;   // Gradient x
                pixels[offset + 1] = m_gradient4[2 * (value & 0x0F)][1] * 64 + 64; // Gradient y
                pixels[offset + 2] = m_gradient4[2 * (value & 0x0F)][2] * 64 + 64; // Gradient z
                pixels[offset + 3] = value;                     // Permuted index
            }

        m_permTexture.initWithData2D(pixels, 256, 256);
    }

    m_simplexTexture.initWithData1D(m_simplex4, 64);

    {
		std::array<char, 256 * 256 * 4> pixels;
        for (int i = 0; i<256; i++)
            for (int j = 0; j<256; j++) 
            {
                int offset = (i * 256 + j) * 4;
                char value = m_perm[(j + m_perm[i]) & 0xFF];
                pixels[offset] = m_gradient4[value & 0x1F][0] * 64 + 64;   // Gradient x
                pixels[offset + 1] = m_gradient4[value & 0x1F][1] * 64 + 64; // Gradient y
                pixels[offset + 2] = m_gradient4[value & 0x1F][2] * 64 + 64; // Gradient z
                pixels[offset + 3] = m_gradient4[value & 0x1F][3] * 64 + 64; // Gradient z
            }

        m_gradientTexture.initWithData2D(pixels, 256, 256);
    }

    m_noiseSSBO.setStorage(std::array<GpuNoiseInfo, 1>{ {m_permTexture.generateHandle(), m_simplexTexture.generateHandle(), m_gradientTexture.generateHandle(), m_time, m_densityFactor, m_densityHeight, m_noiseScale, m_noiseSpeed}}, GL_DYNAMIC_STORAGE_BIT);
}

void SimplexNoise::bindNoiseBuffer(BufferBindings::Binding binding) const
{
    m_noiseSSBO.bindBase(binding);
}

Buffer& SimplexNoise::getNoiseBuffer()
{
    return m_noiseSSBO;
}

bool SimplexNoise::showNoiseGUI()
{
    ImGui::Begin("Noise GUI");
    const bool noiseChanged = showNoiseGUIContent();
    ImGui::End();

    return noiseChanged;
}

bool SimplexNoise::showNoiseGUIContent()
{
    bool res = false;
    ImGui::Text("Density and Noise Settings");
    ImGui::Separator();
    if (ImGui::SliderFloat("Noise Scale", &m_noiseScale, 0.0f, 0.1f))
    {
        getNoiseBuffer().setContentSubData(m_noiseScale, offsetof(GpuNoiseInfo, noiseScale)); 
        res = true;
    }        
    if (ImGui::SliderFloat("Noise Speed", &m_noiseSpeed, 0.0f, 1.0f))
    {
        getNoiseBuffer().setContentSubData(m_noiseSpeed, offsetof(GpuNoiseInfo, noiseSpeed)); 
        res = true;
    }
    if (ImGui::SliderFloat("Density Factor", &m_densityFactor, 0.0f, 0.2f))
    {
        getNoiseBuffer().setContentSubData(m_densityFactor, offsetof(GpuNoiseInfo, heightDensityFactor)); 
        res = true;
	}
	if (ImGui::SliderFloat("Density Height", &m_densityHeight, 0.0f, 4.0f))
	{
		getNoiseBuffer().setContentSubData(glm::log(m_densityHeight), offsetof(GpuNoiseInfo, heightDensityStart));
		res = true;
	}
    return res;
}
