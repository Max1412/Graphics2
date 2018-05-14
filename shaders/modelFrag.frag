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
    int64_t shadowMap;      // can be sampler2D or samplerCube
    float quadratic;        // spot, point
    float cutOff;           // spot
    float outerCutOff;      // spot
};

layout (std430, binding = LIGHTS_BINDING) readonly buffer LightBuffer
{
    Light lights[];
};

float calculateShadow(in int lightIndex, in vec3 fragPos, in vec3 lightDir)
{
    //transform position to light space
    vec4 fragPosLightSpace = lights[lightIndex].lightSpaceMatrix * vec4(fragPos, 1.0f);

    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;

    // handle sampling outside the shadow mapping "far" border
    if(projCoords.z > 1.0)
        return 0.0;

    vec2 texSize = textureSize(sampler2D(lights[lightIndex].shadowMap), 0);
    // if(any(greaterThan(projCoords.xy, vec2(1.0f))))
    // {
    //     return 0.0;
    // }

    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(sampler2D(lights[lightIndex].shadowMap), projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

    // BIAS: TODO make this selectable
    float bias = 0.0;
    //bias = 0.005;
    // normal and lightDir should be in VIEW SPACE
    //bias = -max(0.011  * (1.0 - dot(normalize(interpNormal), normalize(lightDir))), 0.004);  
    //bias = 0.0025 * tan(acos(clamp(dot(normalize(interpNormal), lightDir), 0.0, 1.0)));
    //bias = 0.0;

    float cos_phi = max( dot( normalize(passNormal), normalize(lightDir)), 0.0f);
    bias = -0.00001 * tan( acos( cos_phi ) );

    float shadow = 0.0;

    // check whether current frag pos is in shadow
    //shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

    // PCF : TODO make this selectable
    // TODO use random samples
    vec2 texelSize = 1.0 / textureSize(sampler2D(lights[lightIndex].shadowMap), 0);
    int kernelSize = 13; // TODO make this selectable
    int go = kernelSize / 2;
    for(int x = -go; x <= go; ++x)
    {
        for(int y = -go; y <= go; ++y)
        {
            float pcfDepth = texture(sampler2D(lights[lightIndex].shadowMap), projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= kernelSize * kernelSize;
    
    return shadow;
}

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

    float ambientFactor = 0.25;
    vec3 ambient = ambientFactor * diffCol; // TODO ambient texture/color from assimp
    vec3 normal = normalize(passNormal);
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
            lightingColor += (diffuse + specular);

            // TODO shadow doesn't even show up, fix it
            float shadowFactor = (1.0f - calculateShadow(i, passFragPos, lightDir));
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
            lightingColor += (diffuse + specular);
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

            float shadowFactor = (1.0f - calculateShadow(i, passFragPos, lightDir));
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
    if(col.a <= 0.9f)
        discard;

    fragColor = col;
}