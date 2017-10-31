#version 430
#extension GL_ARB_bindless_texture : require

in vec3 texCoords;

out vec4 fragColor;

layout(binding = 3, std430) buffer textureBuffer
{
    samplerCube skybox;
};

void main()
{
    fragColor = texture(skybox, texCoords);
}