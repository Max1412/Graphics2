#pragma once

#include "light.glsl"

layout(binding = 0, std430) buffer voxelGridBuffer
{
    sampler3D voxelGrid;
};

vec3 applyVolumetricLightingManual(in vec3 colorWithoutVolumetric)
{
    float originalZ = gl_FragCoord.z / gl_FragCoord.w;
    vec3 texCoord = vec3(gl_FragCoord.xy / vec2(1600.0, 900.0), originalZ / 10.0f); //TODO: hack! find the correct z-value!!
    vec4 texEntry = texture(voxelGrid, texCoord);
    return colorWithoutVolumetric * texEntry.w + texEntry.xyz;
}

// CAUTION: to use this you have to enable blending with glBlendFunc(GL_ONE, GL_SRC_ALPHA)
vec4 getVolumetricLighting()
{
    float originalZ = gl_FragCoord.z / gl_FragCoord.w;
    vec3 texCoord = vec3(gl_FragCoord.xy / vec2(1600, 900), originalZ);
    return texture(voxelGrid, texCoord);
}
