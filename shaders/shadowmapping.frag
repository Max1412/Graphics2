#version 430
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require

in vec3 passPosition;
in vec3 interpNormal;

struct Light
{
    vec3 position;
    int type; //0 directional, 1 point light, 2 spot light
    vec3 color;
    float spotCutoff;
    vec3 spotDirection;
    float spotExponent;
    mat4 lightSpaceMatrix;
    uint64_t shadowMap; //can be sampler2D or samplerCube
};

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

layout (std430, binding = LIGHTS_BINDING) restrict readonly buffer LightBuffer
{
    Light light[];
};

layout (std430, binding = MATERIAL_BINDING) restrict readonly buffer MaterialBuffer
{
    Material material[];
};

// Shadow Mapping //////////////
in vec4 fragPosLightspace;
//layout(binding = 7, std430) buffer ShadowMapBuffer
//{
//    sampler2D ShadowMapTexture;
//};

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform vec3 lightAmbient;
uniform int matIndex;

layout( location = 0 ) out vec4 fragmentColor;


float CalculateShadow(in vec4 fragPosLightSpace, in vec3 lightDir)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;

    // handle sampling outside the shadow mapping "far" border
    if(projCoords.z > 1.0)
        return 0.0;

    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(sampler2D(light[0].shadowMap), projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

    // BIAS: TODO make this selectable
    float bias = 0.0;
    //bias = 0.005;
    // normal and lightDir should be in VIEW SPACE
    //bias = -max(0.011  * (1.0 - dot(normalize(interpNormal), normalize(lightDir))), 0.004);  
    //bias = 0.0025 * tan(acos(clamp(dot(normalize(interpNormal), lightDir), 0.0, 1.0)));
    //bias = 0.0;

    float cos_phi = max( dot( normalize(interpNormal), normalize(lightDir)), 0.0f);
    bias = -0.001 * tan( acos( cos_phi ) );

    float shadow = 0.0;

    // check whether current frag pos is in shadow
    //shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

    // PCF : TODO make this selectable
    // TODO use random samples
    vec2 texelSize = 1.0 / textureSize(sampler2D(light[0].shadowMap), 0);
    int kernelSize = 13; // TODO make this selectable
    int go = kernelSize / 2;
    for(int x = -go; x <= go; ++x)
    {
        for(int y = -go; y <= go; ++y)
        {
            float pcfDepth = texture(sampler2D(light[0].shadowMap), projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= kernelSize * kernelSize;
    
    return shadow;
}


void main() 
{
    vec3 passNormal = vec3(0.0, 0.0, 0.0);
    passNormal = interpNormal;
    passNormal = normalize( passNormal );
    vec3 lightVector;
    float spot;
    vec3 diffuse_color;
    float diffuse_alpha;
    Material mat = material[ matIndex];
    diffuse_color = mat.diffColor;
    diffuse_alpha = 1.f - mat.kt;

    fragmentColor.rgb = mat.kd*diffuse_color*lightAmbient;


    for ( int i = 0; i < light.length(); i++) 
    {
        vec4 pos = vec4(light[i].position, light[i].type > 0 ? 1.0f : 0.0f);
        vec3 light_camcoord = (ViewMatrix * pos).xyz;
        if (pos.w > 0.001f)
            lightVector = normalize( light_camcoord - passPosition);
        else
            lightVector = normalize(light_camcoord);
        float cos_phi = max( dot( passNormal, lightVector), 0.000001f);

        vec3 eye = normalize( -passPosition);
        vec3 reflection = normalize( reflect( -lightVector, passNormal));
        float cos_psi_n = pow( max( dot( reflection, eye), 0.000001f), mat.shininess);

        if (light[i].spotCutoff < 0.001f)
            spot = 1.0;
        else 
        {
            float cos_phi_spot = max( dot( -lightVector, normalize(mat3(ViewMatrix) * light[i].spotDirection)), 0.000001f);
            if( cos_phi_spot >= cos( light[i].spotCutoff))
                spot = pow( cos_phi_spot, light[i].spotExponent);
            else
                spot = 0.0f;
        }
        fragmentColor.rgb += mat.kd * spot * diffuse_color * cos_phi * light[i].color;
        fragmentColor.rgb += mat.ks * spot * mat.specColor * cos_psi_n * light[i].color;
    }
    
    float ambient = 0.15;
    float shadowFactor = ambient + (1 - CalculateShadow(fragPosLightspace, lightVector));
    fragmentColor.rgb *= shadowFactor;

    fragmentColor.a = diffuse_alpha;
}