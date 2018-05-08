#version 430 

layout (location = VERTEX_LAYOUT) in vec3 vertexPosition;
layout (location = NORMAL_LAYOUT) in vec3 vertexNormal;
layout (location = TEXCOORD_LAYOUT) in vec3 vertexTexCoord;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

uniform int meshIndex;

out vec3 passNormal;
out vec3 passTexCoord;
out vec3 passFragPos;

layout (std430, binding = MODELMATRICES_BINDING) buffer ModelMatrixBuffer
{
    mat4 modelMatrices[];
};

void main()
{
    mat4 modelMatrix = modelMatrices[meshIndex];
    mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;
    gl_Position = mvp * vec4(vertexPosition, 1.0f);
    passNormal = vec3(transpose(inverse(modelMatrix)) * vec4(vertexNormal, 0.0f));
    passTexCoord = vertexTexCoord;
    passFragPos = vec3(modelMatrix * vec4(vertexPosition, 1.0f));
}