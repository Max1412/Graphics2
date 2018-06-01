#version 460
layout (location = 0) in vec3 vertexPosition;

uniform mat4 lightSpaceMatrix;
uniform mat4 ModelMatrix = mat4(1.0f);

layout(std430, binding = MODELMATRICES_BINDING) buffer ModelMatrixBuffer
{
    mat4 modelMatrices[];
};

subroutine mat4 getModelMatrix();

layout(index = 0) subroutine(getModelMatrix) mat4 bufferModelMatrix()
{
    return modelMatrices[gl_DrawID];
}

layout(index = 1) subroutine(getModelMatrix) mat4 uniformModelMatrix()
{
    return ModelMatrix;
}

layout(location = 0) subroutine uniform getModelMatrix drawMode;

void main()
{
    mat4 modelMatrix = drawMode();
    gl_Position = lightSpaceMatrix * modelMatrix * vec4(vertexPosition, 1.0);
}