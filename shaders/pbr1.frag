#version 430

layout (location = 0) out vec4 fragColor;

in vec3 passPosition;
in vec3 Normal;

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

uniform int matIndex;

uniform vec3 camPos;

const float PI = 3.14159265359;

struct PBRMaterial
{
    vec4 baseColor;
    float F0;
    float metalness;
    float roughness;
};

struct Light
{
    vec4 position;
    vec4 color;
};

layout (std430, binding = 1) restrict readonly buffer MaterialBuffer {
    PBRMaterial materials[];
};

layout (std430, binding = 2) restrict readonly buffer LightBuffer {
    Light lights[];
};

// Schlick approximation of the fresnel term
// calculates the ratio between reflection and refraction of a surface
// based on the viewing angle cosTheta. F0 is the "base reflectivity"
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
} 

// Trowbridge-Reitz GGX
// statistically approximates the ratio of microfacets aligned
// to the halway vector H
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / denom;
}


// Schlick geometry function
// statistically approximates the ratio of microfacets that overshadow each other
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

// Smiths Geometry method
// takes into account:
//      view direction (geometry obstruction)
//      light direction (geometry shadowing)
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

void main()
{
    PBRMaterial currentMaterial = materials[matIndex];
    vec3 albedo = currentMaterial.baseColor.rgb;
    float metallic = currentMaterial.metalness;
    float roughness = currentMaterial.roughness;
    float ao = 1.0f;
    vec3 WorldPos = passPosition;

    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - WorldPos);

    vec3 F0 = vec3(currentMaterial.F0); 
    F0 = mix(F0, albedo, metallic);
               
    // reflectance equation
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < lights.length(); ++i) 
    {
        Light currentLight = lights[i];
        vec3 lightPosition = currentLight.position.xyz;
        vec3 lightColor = currentLight.color.rgb;

        // calculate per-light radiance
        vec3 L = normalize(lightPosition - WorldPos);
        vec3 H = normalize(V + L);
        float distance    = length(lightPosition - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance     = lightColor * attenuation;        
        
        // calculate parts of the Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);        
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;      
        
        // Cook-Torrance specular BRDF term: DFG / 4(w0 * n)(wi * n)
        vec3 nominator    = NDF * G * F;
        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        vec3 specular     = nominator / denominator;
            
        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);                
        Lo += (kD * albedo / PI + specular) * radiance * NdotL; 
    }   
  
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
    
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));  
   
    fragColor = vec4(color, 1.0);
}