#version 430
#extension GL_ARB_bindless_texture : require

in vec2 passTexCoord;

out vec4 fragColor;

uniform vec2 dimensions;

uniform int useGrayscale;
uniform int useBlur;
uniform int useSharpen;


layout(binding = 6, std430) buffer textureBuffer
{
    sampler2D inputTexture;
};

vec4 computeGreyscale(in vec4 color)
{
    float grey = 0.21 * color.r + 0.72 * color.g + 0.07 * color.b;
    return vec4(grey, grey, grey, color.a);
}

vec3 applyKernel3x3(in float kernel[9], in float factor)
{
    vec2 offset = vec2(1.0) / dimensions;
    vec2 imageOffsets[9] = vec2[](
            vec2(-offset.x,  offset.y), // top-left
            vec2( 0.0f,     offset.y), // top-center
            vec2( offset.x,  offset.y), // top-right
            vec2(-offset.x,  0.0f),   // center-left
            vec2( 0.0f,     0.0f),   // center-center
            vec2( offset.x,  0.0f),   // center-right
            vec2(-offset.x,  -offset.y), // bottom-left
            vec2( 0.0f,     -offset.y), // bottom-center
            vec2( offset.x,  -offset.y)  // bottom-right    
    );
    vec3 col = vec3(0.0);
    for(int i = 0; i < 9; i++)
    {
        col += factor * kernel[i] * vec3(texture(inputTexture, passTexCoord + imageOffsets[i]));
    }
    return col;
}

void main() {
    vec4 currentColor = texture(inputTexture, passTexCoord);

    if(useSharpen == 1)
    {
        float kernel[9] = float[](
                -1, -1, -1,
                -1,  5, -1,
                -1, -1, -1
        );
        currentColor.rgb = applyKernel3x3(kernel, 1.0);
    }

    if(useBlur == 1)
    {
        float kernel[9] = float[](
                1, 2, 1,
                2, 4, 2,
                1, 2, 1
        );
        currentColor.rgb = applyKernel3x3(kernel, 1.0 / 16.0);
    }

    if(useGrayscale == 1)
    {
        currentColor = computeGreyscale(currentColor);
    }

    fragColor = currentColor;
}