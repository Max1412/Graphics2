#version 430
#extension GL_ARB_bindless_texture : require

in vec3 passPositionView;
in vec3 passPositionWorld;
in vec3 interpNormal;

#include "common/light.glsl"
#include "common/shadowMapping.glsl"

struct Material
{
    vec3 diffColor;
    float kd;
    vec3 specColor;
    float ks;
    float shininess;
    float kt;
    int reflective;
};

layout (std430, binding = MATERIAL_BINDING) restrict readonly buffer MaterialBuffer
{
    Material material[];
};

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform vec3 lightAmbient;
uniform int matIndex;

uniform vec3 camPos;

layout( location = 0 ) out vec4 fragmentColor;

void main() 
{
    vec3 passNormal = normalize(interpNormal);
    Material mat = material[ matIndex];
    vec3 diffuse_color = mat.diffColor;
    float diffuse_alpha = 1.f - mat.kt;

    vec3 viewDir = normalize(camPos - passPositionWorld);

    fragmentColor.rgb = mat.kd*diffuse_color*lightAmbient;

    float ambient = 0.15;

    for (int i = 0; i < lights.length(); i++)
    {
        LightResult lRes = getLight(i, passPositionWorld, viewDir, passNormal, mat.shininess);
        float shadowFactor = calculateShadowPCF(i, passPositionWorld, passNormal, lRes.direction);
        fragmentColor.rgb += shadowFactor * (diffuse_color * lRes.diffuse + mat.specColor * lRes.specular);
    }
    
    fragmentColor.a = diffuse_alpha;
}