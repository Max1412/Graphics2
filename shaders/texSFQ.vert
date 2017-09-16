#version 430

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texCoord;

out vec2 passTexCoord;

void main() {
    passTexCoord = texCoord;
    gl_Position = vec4(position);
}