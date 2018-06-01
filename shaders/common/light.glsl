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
    uvec2 shadowMap;        // can be sampler2D or samplerCube
    float quadratic;        // spot, point
    float cutOff;           // spot
    float outerCutOff;      // spot
    int pcfKernelSize;
};

layout(std430, binding = LIGHTS_BINDING) readonly buffer LightBuffer
{
    Light lights[];
};

uniform vec3 ambient = vec3(0.15f);

struct LightResult
{
    vec3 diffuse;
    vec3 specular;
    vec3 direction;
};

LightResult getLight(int lightIndex, vec3 worldPos, vec3 viewDir, vec3 normal, float specExponent)
{
    LightResult result;

    Light currentLight = lights[lightIndex];
    if (currentLight.type == 0) // D I R E C T I O N A L
    {
        vec3 lightDir = normalize(-currentLight.direction);

        // diffuse shading
        float diff = max(dot(normal, lightDir), 0.0);

        // specular shading
        vec3 halfwayDir = normalize(lightDir + viewDir);  
        float spec = pow(max(dot(normal, halfwayDir), 0.0), specExponent);

        // combine results
        result.diffuse = currentLight.color * diff;
        result.specular = currentLight.color * spec;
        result.direction = lightDir;

        //float shadowFactor = calculateShadowPCF(lightIndex, worldPos, passNormal, lightDir);
        //vec3 thisLight = shadowFactor * (diffuse + specular);
        //lightingColor += (diffuse + specular);
    }
    if (currentLight.type == 1) // P O I N T
    {
        vec3 lightDir = normalize(currentLight.position - worldPos);

        // diffuse shading
        float diff = max(dot(normal, lightDir), 0.0);

        // specular shading
        vec3 halfwayDir = normalize(lightDir + viewDir);  
        float spec = pow(max(dot(normal, halfwayDir), 0.0), specExponent);

        // attenuation
        float distance = length(currentLight.position - worldPos);
        float attenuation = 1.0 / max(0.001f, (currentLight.constant + currentLight.linear * distance + currentLight.quadratic * (distance * distance)));

        // combine results
        result.diffuse = currentLight.color * diff * attenuation;
        result.specular = currentLight.color * spec * attenuation;
        result.direction = lightDir;

        //float shadowFactor = (1.0f - calculateCubeShadow(lightIndex, worldPos, lightDir));
        //vec3 thisLight = shadowFactor * (diffuse + specular);
        //lightingColor += (diffuse + specular);
    }
    if (currentLight.type == 2) // S P O T
    {
        vec3 lightDir = normalize(currentLight.position - worldPos);

        // diffuse shading
        float diff = max(dot(normal, lightDir), 0.0);

        // specular shading
        vec3 halfwayDir = normalize(lightDir + viewDir);  
        float spec = pow(max(dot(normal, halfwayDir), 0.0), specExponent);

        // attenuation
        float distance = length(currentLight.position - worldPos);
        float attenuation = 1.0 / max(0.001f, (currentLight.constant + currentLight.linear * distance + currentLight.quadratic * (distance * distance)));

        // spotlight intensity
        float theta = dot(lightDir, normalize(-currentLight.direction));
        float epsilon = currentLight.cutOff - currentLight.outerCutOff;
        float intensity = clamp((theta - currentLight.outerCutOff) / epsilon, 0.0, 1.0);

        // combine results
        result.diffuse = currentLight.color * diff * attenuation * intensity;
        result.specular = currentLight.color * spec * attenuation * intensity;
        result.direction = lightDir;

        //float shadowFactor = calculateShadowPCF(lightIndex, worldPos, passNormal, lightDir);
        //vec3 thisLight = shadowFactor * (diffuse + specular);
        //lightingColor += (diffuse + specular);
    }
    return result;
}
