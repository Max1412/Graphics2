#version 460
#extension GL_ARB_bindless_texture : require
//#extension GL_ARB_gpu_shader_int64 : require
layout(early_fragment_tests) in;

uniform vec3 cameraPos;
uniform mat4 viewMatrix;

layout(location = 0) in vec3 passFragPos;
layout(location = 1) in vec3 passTexCoord;
layout(location = 2) in vec3 passNormal;
layout(location = 3) flat in uint passDrawID;
layout(location = 4) in vec3 tangent;
layout(location = 5) in vec3 bitangent;

#include "common/light.glsl"
#include "common/material.glsl"
#include "common/shadowMapping.glsl"

out vec4 fragColor;

void main()
{
    uint matIndex = materialIndices[passDrawID];
    Material currentMaterial = materials[matIndex];

    vec3 diffCol = vec3(1.0f);
    vec3 specCol = vec3(1.0f);

    // diffuse color or texture
    if(currentMaterial.diffColor.a != 0.0f)
        diffCol = texture(currentMaterial.diffTexture, passTexCoord.rg).rgb;
    else
        diffCol = currentMaterial.diffColor.rgb;

    // specular color or texture TODO USE THIS
    if(currentMaterial.specColor.a != 0.0f)
        specCol = texture(currentMaterial.specTexture, passTexCoord.rg).rgb;
    else
        specCol = currentMaterial.specColor.rgb;

    float ambientFactor = 0.15;
    vec3 ambient = ambientFactor * diffCol; // TODO ambient texture/color from assimp

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

    vec3 viewDir = normalize(cameraPos - passFragPos);

    vec3 lightingColor = vec3(0.0f);
    lightingColor += ambient;

    for(int i = 0; i < lights.length(); i++)
    {
        Light currentLight = lights[i];
        if(currentLight.type == 0) // D I R E C T I O N A L
        {
            vec3 lightDir = normalize(-currentLight.direction);

            // diffuse shading
            float diff = max(dot(normal, lightDir), 0.0);

            // specular shading
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), currentMaterial.Ns);

            // combine results
            vec3 diffuse = currentLight.color * diff * diffCol;
            vec3 specular = currentLight.color * spec * specCol;

            float shadowFactor = calculateShadowPCF(i, passFragPos, passNormal, lightDir);
            vec3 thisLight = shadowFactor * (diffuse + specular);
            lightingColor += thisLight;
        }
        if(currentLight.type == 1) // P O I N T
        {
            vec3 lightDir = normalize(currentLight.position - passFragPos);

            // diffuse shading
            float diff = max(dot(normal, lightDir), 0.0);

            // specular shading
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), currentMaterial.Ns);

            // attenuation
            float distance = length(currentLight.position - passFragPos);
            float attenuation = 1.0 / max(0.001f, (currentLight.constant + currentLight.linear * distance + currentLight.quadratic * (distance * distance)));    

            // combine results
            vec3 diffuse = currentLight.color * diff * diffCol;
            vec3 specular = currentLight.color * spec * specCol;
            diffuse *= attenuation;
            specular *= attenuation;

            float shadowFactor = (1.0f - calculateCubeShadow(i, passFragPos, lightDir));
            vec3 thisLight = shadowFactor * (diffuse + specular);
            lightingColor += thisLight;
        }
        if(currentLight.type == 2) // S P O T
        {
            vec3 lightDir = normalize(currentLight.position - passFragPos);

            // diffuse shading
            float diff = max(dot(normal, lightDir), 0.0);

            // specular shading
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), currentMaterial.Ns);

            // attenuation
            float distance = length(currentLight.position - passFragPos);
            float attenuation = 1.0 / max(0.001f, (currentLight.constant + currentLight.linear * distance + currentLight.quadratic * (distance * distance)));    

            // spotlight intensity
            float theta = dot(lightDir, normalize(-currentLight.direction)); 
            float epsilon = currentLight.cutOff - currentLight.outerCutOff;
            float intensity = clamp((theta - currentLight.outerCutOff) / epsilon, 0.0, 1.0);

            // combine results
            vec3 diffuse = currentLight.color * diff * diffCol;
            vec3 specular = currentLight.color * spec * specCol;
            diffuse *= attenuation * intensity;
            specular *= attenuation * intensity;

            float shadowFactor = calculateShadowPCF(i, passFragPos, passNormal, lightDir);
            vec3 thisLight = shadowFactor * (diffuse + specular);
            lightingColor += thisLight;
        }
    }
    vec4 col = vec4(lightingColor, 1.0);

    if(currentMaterial.opacity == -1.0f) // has opacity texture instead of opacity
        col.a = texture(currentMaterial.opacityTexture, passTexCoord.rg).r;
    else
        col.a = currentMaterial.opacity;

    // SPONZA HACK
    // if(col.a <= 0.9f)
    //     discard;

    fragColor = col;
}