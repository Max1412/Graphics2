#version 430
#extension GL_ARB_bindless_texture : require

out vec4 fragColor;

in vec3 localPos;

layout(binding = 5, std430) buffer textureBuffer
{
    sampler2D equirectangularMap;
};

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

void main()
{        
    vec2 uv = SampleSphericalMap(normalize(localPos)); // make sure to normalize localPos
    vec3 color = texture(equirectangularMap, uv).rgb;
    
    fragColor = vec4(color, 1.0);
}