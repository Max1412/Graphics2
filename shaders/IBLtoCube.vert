#version 430

layout (location = 0) in vec3 vertexPosition;

out vec3 localPos;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    localPos = vertexPosition;  
    gl_Position =  projection * view * vec4(localPos, 1.0);
}