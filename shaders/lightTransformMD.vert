#version 460
layout (location = 0) in vec3 vertexPosition;

uniform mat4 lightSpaceMatrix;

layout (std430, binding = MODELMATRICES_BINDING) buffer ModelMatrixBuffer
{
    mat4 modelMatrices[];
};

void main()
{
    mat4 modelMatrix = modelMatrices[gl_DrawID];
    gl_Position = lightSpaceMatrix * modelMatrix * vec4(vertexPosition, 1.0);
}  