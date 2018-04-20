#version 430

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

out vec2 passTexCoord;

void main() 
{
    passTexCoord = texCoord;
    gl_Position = vec4(position, 0.0f, 1.0f);
}