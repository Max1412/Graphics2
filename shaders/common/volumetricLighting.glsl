#pragma once

#include "light.glsl"

layout(bindless_sampler) uniform sampler3D voxelGrid;
uniform float maxRange;
uniform vec2 screenRes;

vec3 applyVolumetricLightingManual(in vec3 colorWithoutVolumetric, in float viewZ)
{
    float zDist = -viewZ / maxRange;
    zDist /= exp(-1.f+zDist); //use exponential depth
    vec3 texCoord = vec3(gl_FragCoord.xy / screenRes, zDist);
    vec4 texEntry = texture(voxelGrid, texCoord);
    return colorWithoutVolumetric * texEntry.w + texEntry.xyz;
}

// CAUTION: to use this you have to enable blending with glBlendFunc(GL_ONE, GL_SRC_ALPHA)
vec4 getVolumetricLighting(in float viewZ)
{
    float zDist = -viewZ / maxRange;
    zDist /= exp(-1.f + zDist); //use exponential depth
    vec3 texCoord = vec3(gl_FragCoord.xy / screenRes, zDist);
    return texture(voxelGrid, texCoord);
}
