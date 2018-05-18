#include "SimplexNoise.h"

SimplexNoise::SimplexNoise()
{
    m_time = static_cast<float>(glfwGetTime());

    {
        char *pixels;
        pixels = (char*)malloc(256 * 256 * 4);
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

    m_simplexTexture.initWithData1D(m_simplex4.data(), 64);

    {
        char *pixels;
        pixels = (char*)malloc(256 * 256 * 4);
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

    m_noiseSSBO.setStorage(std::array<GpuNoiseInfo, 1>{ {m_permTexture.generateHandle(), m_simplexTexture.generateHandle(), m_gradientTexture.generateHandle(), m_time, m_densityFactor, m_noiseScale, m_noiseSpeed}}, GL_DYNAMIC_STORAGE_BIT);
}

void SimplexNoise::bindNoiseBuffer(BufferBindings::Binding binding) const
{
    m_noiseSSBO.bindBase(binding);
}

Buffer& SimplexNoise::getNoiseBuffer()
{
    return m_noiseSSBO;
}
