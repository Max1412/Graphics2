#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;

out vec4 fragColor;

layout(binding = 6, std430) buffer textureBuffer
{
    sampler2D inputTexture;
};

void main() {
    vec4 inputTexPixel = texture(inputTexture, passTexCoord);
    fragColor = inputTexPixel;
}