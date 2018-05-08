#version 430
#extension GL_ARB_bindless_texture : require

uniform int materialIndex;

in vec3 passNormal;
in vec3 passTexCoord;

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

layout (std430, binding = MATERIAL_BINDING) buffer MaterialBuffer
{
    Material materials[];
};

void main()
{
    Material currentMaterial = materials[materialIndex];

    vec4 col = vec4(1.0f);
    vec4 spec = vec4(1.0f);

    if(currentMaterial.diffColor.a != 0.0f)
        col = texture(currentMaterial.diffTexture, passTexCoord.rg);
    else
        col = vec4(currentMaterial.diffColor.rgb, 1.0f);

    if(currentMaterial.specColor.a != 0.0f)
        spec = texture(currentMaterial.specTexture, passTexCoord.rg);
    else
        spec = vec4(currentMaterial.specColor.rgb, 1.0f);

    if(currentMaterial.opacity == -1.0f) // has opacity texture instead of opacity
        col.a = texture(currentMaterial.opacityTexture, passTexCoord.rg).r;
    else
        col.a = currentMaterial.opacity;

    fragColor = col;//vec4(currentMaterial.diffColor.rgb, 1.0);
}