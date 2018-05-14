#pragma once

struct Light
{
    mat4 lightSpaceMatrix;
    vec3 color;             // all
    int type;               // 0 directional, 1 point light, 2 spot light
    vec3 position;          // spot, point
    float constant;         // spot, point
    vec3 direction;         // dir, spot
    float linear;           // spot, point
    uvec2 shadowMap;      // can be sampler2D or samplerCube
    float quadratic;        // spot, point
    float cutOff;           // spot
    float outerCutOff;      // spot
};

layout(std430, binding = LIGHTS_BINDING) readonly buffer LightBuffer
{
    Light lights[];
};
