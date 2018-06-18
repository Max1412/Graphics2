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
    sampler2D bumpTexture;
    int bumpType; //0 = no bump, 1 = normal map, 2 = height map
};

layout(std430, binding = MATERIAL_BINDING) readonly buffer MaterialBuffer
{
    Material materials[];
};

layout(std430, binding = MATERIAL_INDICES_BINDING) readonly buffer MaterialIndexBuffer
{
    uint materialIndices[];
};

vec3 getDiffColor(int materialIndex)
{
    Material currentMaterial = materials[materialIndex];
    // diffuse color or texture
    if (currentMaterial.diffColor.a != 0.0f)
        return texture(currentMaterial.diffTexture, passTexCoord.rg).rgb;
    else
        return currentMaterial.diffColor.rgb;
}

vec3 getSpecColor(int materialIndex)
{
    Material currentMaterial = materials[materialIndex];
    // specular color or texture TODO USE THIS
    if (currentMaterial.specColor.a != 0.0f)
        return texture(currentMaterial.specTexture, passTexCoord.rg).rgb;
    else
        return currentMaterial.specColor.rgb;
}
