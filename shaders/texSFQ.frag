#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;
//layout(pixel_center_integer​) in vec4 gl_FragCoord;

out vec4 fragColor;

layout(binding = 0, std430) buffer textureBuffer
{
    sampler2D inputTexture;
};

layout(binding = 1, std430) buffer immageSSBO
{
    layout(r8ui) uimage2D image;
};

void main() {
    //vec2 texc = vec2(gl_FragCoord.x/1920.0, gl_FragCoord.y/1080.0);
    vec4 inputTexPixel = texture(inputTexture, passTexCoord);
    ivec2 actualCoords = ivec2(gl_FragCoord.xy);
    imageStore(image, actualCoords, uvec4(1));
    fragColor = inputTexPixel;//vec4(passTexCoord, 0.0, 1.0);
}