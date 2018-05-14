#version 430

#include "common/light.glsl"

out vec3 posColor;
out int vertexID;

void main()
{
    Light currentLight = lights[gl_VertexID];
    gl_Position = vec4(currentLight.position, 1.0f);
    posColor = currentLight.color;
    vertexID = gl_VertexID;
}