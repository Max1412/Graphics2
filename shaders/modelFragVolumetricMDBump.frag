#version 430
#extension GL_ARB_bindless_texture : require
//#extension GL_ARB_gpu_shader_int64 : require
layout(early_fragment_tests) in;

layout(binding = CAMERA_BINDING, std430) buffer cameraBuffer
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 camPos;
};

layout(location = 0) in vec3 passWorldPos;
layout(location = 1) in vec3 passTexCoord;
layout(location = 2) in vec3 passNormal;
layout(location = 3) in vec3 passViewPos;
layout(location = 4) flat in uint passDrawID;
layout(location = 5) in vec3 tangent;
layout(location = 6) in vec3 bitangent;

#include "common/light.glsl"
#include "common/material.glsl"
#include "common/shadowMapping.glsl"
#include "common/volumetricLighting.glsl"

out vec4 fragColor;

void main()
{
    int materialIndex = int(materialIndices[passDrawID]);
    Material currentMaterial = materials[materialIndex];

    vec3 diffCol = getDiffColor(materialIndex);
    vec3 specCol = getSpecColor(materialIndex);

    vec3 viewDir = normalize(camPos - passWorldPos);

	vec3 normal;
	if(currentMaterial.bumpType == 1)
	{
		mat3 TBN = mat3(tangent, bitangent, passNormal);
		normal = texture(currentMaterial.bumpTexture, passTexCoord.rg).rgb;
		normal = normalize(normal * 2.0 - 1.0);   
		normal = normalize(TBN * normal);
	}
	else if(currentMaterial.bumpType == 2)
	{
		vec2 size = vec2(0.5,0.0); //"strength" of bump-mapping
		ivec3 off = ivec3(-1,0,1);

		float s01 = textureOffset(currentMaterial.bumpTexture, passTexCoord.rg, off.xy).x;
		float s21 = textureOffset(currentMaterial.bumpTexture, passTexCoord.rg, off.zy).x;
		float s10 = textureOffset(currentMaterial.bumpTexture, passTexCoord.rg, off.yx).x;
		float s12 = textureOffset(currentMaterial.bumpTexture, passTexCoord.rg, off.yz).x;
		vec3 va = normalize(vec3(size.xy,s21-s01));
		vec3 vb = normalize(vec3(size.yx,s12-s10));
		vec3 tanSpaceNormal = cross(va,vb);

		mat3 TBN = mat3(tangent, bitangent, passNormal);
		normal = normalize(TBN * tanSpaceNormal);
	}
	else
	{
		normal = normalize(passNormal);
	}

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
    else if (currentMaterial.opacity == -2.0f)
        col.a = getDiffTextureAlpha(materialIndex);
    else
        col.a = currentMaterial.opacity;

    fragColor = col;
}