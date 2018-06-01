#version 460 

layout (location = VERTEX_LAYOUT) in vec3 vertexPosition;
layout (location = NORMAL_LAYOUT) in vec3 vertexNormal;
layout (location = TEXCOORD_LAYOUT) in vec3 vertexTexCoord;

layout(binding = CAMERA_BINDING, std430) buffer cameraBuffer
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 camPos;
};

out vec3 passNormal;
out vec3 passTexCoord;
out vec3 passWorldPos;
out vec3 passViewPos;
flat out uint passDrawID;

layout (std430, binding = MODELMATRICES_BINDING) buffer ModelMatrixBuffer
{
    mat4 modelMatrices[];
};

void main()
{
    mat4 modelMatrix = modelMatrices[gl_DrawID];
    passDrawID = gl_DrawID;

    vec4 worldPos = modelMatrix * vec4(vertexPosition, 1.0f);
    passWorldPos = worldPos.xyz;

    vec4 viewPos = viewMatrix * worldPos;
    passViewPos = viewPos.xyz;

    vec4 projPos = projectionMatrix * viewPos;
    gl_Position = projPos;

    passNormal = mat3(transpose(inverse(modelMatrix))) * vertexNormal;
    passTexCoord = vertexTexCoord;
}