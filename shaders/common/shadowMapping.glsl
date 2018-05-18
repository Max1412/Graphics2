#pragma once

#include "light.glsl"

// cube maps do not work at the moment
float calculateCubeShadow(in int lightIndex, in vec3 worldPos, in vec3 lightDir)
{
    Light currentLight = lights[lightIndex];
    vec3 fragToLight = worldPos - currentLight.position;
    float shadowDepth = texture(samplerCube(currentLight.shadowMap), fragToLight).r;
    shadowDepth *= 1000.0f;
    float currentDepth = length(fragToLight);
    float bias = 0.05f;
    float shadow = currentDepth - bias > shadowDepth ? 1.0f : 0.0f;

    return shadow;
}

float calculateShadow(in int lightIndex, in vec3 worldPos, in vec3 worldNormal, in vec3 lightDir)
{
    //transform position to light space
    vec4 worldPosLightSpace = lights[lightIndex].lightSpaceMatrix * vec4(worldPos, 1.0f);

    // perform perspective divide
    vec3 projCoords = worldPosLightSpace.xyz / worldPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5f + 0.5f;

    // handle sampling outside the shadow mapping "far" border
    if (projCoords.z > 1.0f)
        return 0.0f;

    // get closest depth value from light's perspective (using [0,1] range worldPosLight as coords)
    float closestDepth = texture(sampler2D(lights[lightIndex].shadowMap), projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

    // BIAS: TODO make this selectable
    float bias = 0.0f;
    //bias = 0.005;
    // normal and lightDir should be in VIEW SPACE
    //bias = -max(0.011  * (1.0 - dot(normalize(interpNormal), normalize(lightDir))), 0.004);  
    //bias = 0.0025 * tan(acos(clamp(dot(normalize(interpNormal), lightDir), 0.0, 1.0)));
    //bias = 0.0;

    float cos_phi = max(dot(normalize(worldNormal), normalize(lightDir)), 0.0f);
    bias = -0.00001f * tan(acos(cos_phi));

    float shadow = 0.0f;

    // check whether current frag pos is in shadow
    //shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

    // PCF : TODO make this selectable
    // TODO use random samples
    vec2 texelSize = 1.0f / textureSize(sampler2D(lights[lightIndex].shadowMap), 0);
    int kernelSize = 13; // TODO make this selectable
    int go = kernelSize / 2;
    for (int x = -go; x <= go; ++x)
    {
        for (int y = -go; y <= go; ++y)
        {
            float pcfDepth = texture(sampler2D(lights[lightIndex].shadowMap), projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0f : 0.0f;
        }
    }
    shadow /= kernelSize * kernelSize;

    return shadow;
}
