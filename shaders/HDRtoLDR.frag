#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;

out vec4 fragColor;

uniform float exposure;
uniform float gamma;

layout(bindless_sampler) uniform sampler2D inputTexture;

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x)
{
   return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

void main() 
{
    vec3 hdrColor = texture(inputTexture, passTexCoord).rgb;
    // R E I N H A R D
    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    // Gamma correction 
    mapped = pow(mapped, vec3(1.0 / gamma));
  
    fragColor = vec4(mapped, 1.0f);
    //fragColor = vec4(hdrColor, 1.0+exposure+gamma);

    // U N C H A R T E D 2
    // vec3 curr = Uncharted2Tonemap(exposure * hdrColor);

    // vec3 whiteScale = 1.0f/Uncharted2Tonemap(vec3(W));
    // vec3 color = curr * whiteScale;

    // fragColor = vec4(pow(color, vec3(1.0/gamma)), 1.0f);
}