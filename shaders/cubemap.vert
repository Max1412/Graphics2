#version 430
layout (location = 0) in vec3 vertexPosition;

out vec3 texCoords;

uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;

void main()
{
    texCoords = vertexPosition;
    vec4 pos = ProjectionMatrix * mat4(mat3(ViewMatrix)) * vec4(vertexPosition, 1.0f);
    gl_Position = pos.xyww;
}