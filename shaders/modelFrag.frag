#version 430
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require

uniform int materialIndex;
uniform vec3 cameraPos;

in vec3 passNormal;
in vec3 passTexCoord;
in vec3 passFragPos;

out vec4 fragColor;

struct Material
{
    sampler2D diffTexture;
    sampler2D specTexture;
    sampler2D opacityTexture;
    float opacity;
    float Ns;
    vec4 diffColor;
    vec4 specColor;
    vec4 emissiveColor;
};

layout (std430, binding = MATERIAL_BINDING) readonly buffer MaterialBuffer
{
    Material materials[];
};

struct Light
{
    mat4 lightSpaceMatrix;
    vec3 color;             // all
    int type;               // 0 directional, 1 point light, 2 spot light
    vec3 position;          // spot, point
    float constant;         // spot, point
    vec3 direction;         // dir, spot
    float linear;           // spot, point
    float quadratic;        // spot, point
    float cutOff;           // spot
    float outerCutOff;      // spot
    int64_t shadowMap;      // can be sampler2D or samplerCube
};

layout (std430, binding = LIGHTS_BINDING) readonly buffer LightBuffer
{
    Light lights[];
};

void main()
{
    Material currentMaterial = materials[materialIndex];

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

    vec3 ambient = 0.05 * diffCol; // TODO ambient texture/color from assimp
    vec3 normal = normalize(passNormal);
    vec3 viewDir = normalize(cameraPos - passFragPos);

    // TODO sum up lights
    vec3 lightingColor = vec3(0.0f);
    lightingColor += ambient;

    for(int i = 0; i < lights.length(); i++)
    {
        Light currentLight = lights[i];
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
            float attenuation = 1.0 / (currentLight.constant + currentLight.linear * distance + currentLight.quadratic * (distance * distance));    

            // spotlight intensity
            float theta = dot(lightDir, normalize(-currentLight.direction)); 
            float epsilon = currentLight.cutOff - currentLight.outerCutOff;
            float intensity = clamp((theta - currentLight.outerCutOff) / epsilon, 0.0, 1.0);

            // combine results
            vec3 diffuse = currentLight.color * diff * diffCol;
            vec3 specular = currentLight.color * spec * specCol;
            diffuse *= attenuation * intensity;
            specular *= attenuation * intensity;
            lightingColor += (diffuse + specular);
        }
    }
    
    vec4 col = vec4(lightingColor, 1.0);

    if(currentMaterial.opacity == -1.0f) // has opacity texture instead of opacity
        col.a = texture(currentMaterial.opacityTexture, passTexCoord.rg).r;
    else
        col.a = currentMaterial.opacity;

    fragColor = col;//vec4(currentMaterial.diffColor.rgb, 1.0);
}