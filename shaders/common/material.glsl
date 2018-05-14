#pragma once

struct Material
{
    sampler2D diffTexture;
    sampler2D specTexture;
    sampler2D opacityTexture;
    float opacity;
    float Ns;
    vec4 diffColor;
    vec4 specColor;
    vec4 emissiveColor;
};

layout(std430, binding = MATERIAL_BINDING) readonly buffer MaterialBuffer
{
    Material materials[];
};
