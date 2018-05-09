#version 430
#extension GL_ARB_gpu_shader_int64 : require

struct Light
{
    mat4 lightSpaceMatrix;
    vec3 color;             // all
    int type;               // 0 directional, 1 point light, 2 spot light
    vec3 position;          // spot, point
    float constant;         // spot, point
    vec3 direction;         // dir, spot
    float linear;           // spot, point
    float quadratic;        // spot, point
    float cutOff;           // spot
    float outerCutOff;      // spot
    int64_t shadowMap;      // can be sampler2D or samplerCube
};

layout (std430, binding = LIGHTS_BINDING) readonly buffer LightBuffer
{
    Light lights[];
};

out vec3 posColor;
out int vertexID;

void main()
{
    Light currentLight = lights[gl_VertexID];
    gl_Position = vec4(currentLight.position, 1.0f);
    posColor = currentLight.color;
    vertexID = gl_VertexID;
}