#version 430
#extension GL_ARB_bindless_texture : require
//#extension GL_ARB_gpu_shader_int64 : require

uniform int materialIndex;

layout(binding = CAMERA_BINDING, std430) buffer cameraBuffer
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 camPos;
};

in vec3 passNormal;
in vec3 passTexCoord;
in vec3 passWorldPos;
in vec3 passViewPos;

#include "common/light.glsl"
#include "common/material.glsl"
#include "common/shadowMapping.glsl"
#include "common/volumetricLighting.glsl"

out vec4 fragColor;

void main()
{
    Material currentMaterial = materials[materialIndex];

    vec3 diffCol = getDiffColor(materialIndex);
    vec3 specCol = getSpecColor(materialIndex);

    vec3 normal = normalize(passNormal);
    vec3 viewDir = normalize(camPos - passWorldPos);

    vec3 lightingColor = ambient;

    for (int i = 0; i < lights.length(); i++)
    {
        LightResult lRes = getLight(i, passWorldPos, viewDir, normal, currentMaterial.Ns);
        float shadowFactor = calculateShadowPCF(i, passWorldPos, normal, lRes.direction);
        lightingColor += shadowFactor * (diffCol * lRes.diffuse + specCol * lRes.specular);
    }
    lightingColor = applyVolumetricLightingManual(lightingColor, passViewPos.z);
    vec4 col = vec4(lightingColor, 1.0);

    if (currentMaterial.opacity == -1.0f) // has opacity texture instead of opacity
        col.a = texture(currentMaterial.opacityTexture, passTexCoord.rg).r;
    else
        col.a = currentMaterial.opacity;

    // SPONZA HACK
    if (col.a <= 0.9f)
        discard;

    fragColor = col;
}