#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;
//layout(pixel_center_integer​) in vec4 gl_FragCoord;

out vec4 fragColor;

struct Centroid
{
    vec3 color;
    layout(r8ui) uimage2D image;
};

layout(binding = 0, std430) buffer textureBuffer
{
    sampler2D inputTexture;
};

layout(binding = 1, std430) buffer imageSSBO
{
    Centroid centroids[];
};

void main() {
    //vec2 texc = vec2(gl_FragCoord.x/1920.0, gl_FragCoord.y/1080.0);
    vec4 inputTexPixel = texture(inputTexture, passTexCoord);
    ivec2 actualCoords = ivec2(gl_FragCoord.xy);
    vec3 inputPixel = inputTexPixel.rgb;
    float minimum = 10000;
    for (int i = 0; i < centroids.length(); i++)
    {
        if (abs(length(inputPixel - centroids[i].color)) < minimum)
        {
            imageStore(centroids[i].image, actualCoords, uvec4(1));
        }
    }
    fragColor = inputTexPixel;//vec4(passTexCoord, 0.0, 1.0);
}