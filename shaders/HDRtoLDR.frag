#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;

out vec4 fragColor;

uniform float exposure;
uniform float gamma;

layout(binding = 4, std430) buffer textureBuffer
{
    sampler2D inputTexture;
};

void main() 
{
    vec3 hdrColor = texture(inputTexture, passTexCoord).rgb;
  
    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    // Gamma correction 
    mapped = pow(mapped, vec3(1.0 / gamma));
  
    fragColor = vec4(mapped, 1.0f);//vec4(hdrColor, 1.0+exposure+gamma);//
}