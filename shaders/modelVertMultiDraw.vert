#version 460 

layout (location = VERTEX_LAYOUT) in vec3 vertexPosition;
layout (location = NORMAL_LAYOUT) in vec3 vertexNormal;
layout (location = TEXCOORD_LAYOUT) in vec3 vertexTexCoord;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

layout(location = 0) out vec3 passFragPos;
layout(location = 1) out vec3 passNormal;
layout(location = 2) out vec3 passTexCoord;
layout(location = 3) flat out uint passDrawID;

layout (std430, binding = MODELMATRICES_BINDING) buffer ModelMatrixBuffer
{
    mat4 modelMatrices[];
};

void main()
{
    mat4 modelMatrix = modelMatrices[gl_DrawID];
    passDrawID = gl_DrawID;
    mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;
    gl_Position = mvp * vec4(vertexPosition, 1.0f);
    passNormal = mat3(transpose(inverse(modelMatrix))) * vertexNormal;
    passTexCoord = vertexTexCoord;
    passFragPos = vec3(modelMatrix * vec4(vertexPosition, 1.0f));
}