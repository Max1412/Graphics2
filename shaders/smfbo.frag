#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;

out vec4 fragColor;

layout(binding = 7, std430) buffer textureBuffer
{
    sampler2D inputTexture;
};

void main() {
    float depthValue = texture(inputTexture, passTexCoord).r;
    fragColor = vec4(vec3(depthValue), 1.0);
}