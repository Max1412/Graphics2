#version 430
#extension GL_ARB_bindless_texture : require

#include "common/volumetricLighting.glsl"

in vec3 texCoords;

out vec4 fragColor;

layout(bindless_sampler) uniform samplerCube skybox;

void main()
{
    fragColor = texture(skybox, texCoords);
	fragColor.xyz = applyVolumetricLightingManual(fragColor.xyz, -maxRange);

}