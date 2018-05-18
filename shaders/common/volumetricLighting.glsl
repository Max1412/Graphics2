#pragma once

#include "light.glsl"

layout(binding = 0, std430) buffer voxelGridBuffer
{
    layout(rgba32f) sampler3D voxelGrid;
};

vec3 applyVolumetricLightingManual(in vec3 colorWithoutVolumetric)
{
    float originalZ = gl_FragCoord.z / gl_FragCoord.w;
    vec3 texCoord = vec3(gl_FragCoord.xy, originalZ);
    vec4 texEntry = texture(voxelGrid, texCoord);
    return colorWithoutVolumetric * texEntry.w + texEntry.xyz;
}

// CAUTION: to use this you have to enable blending with glBlendFunc(GL_ONE, GL_SRC_ALPHA)
vec4 getVolumetricLighting()
{
    float originalZ = gl_FragCoord.z / gl_FragCoord.w;
    vec3 texCoord = vec3(gl_FragCoord.xy, originalZ);
    return texture(voxelGrid, texCoord);
}
