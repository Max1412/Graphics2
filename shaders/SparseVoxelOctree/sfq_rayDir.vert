#version 450

layout (location = 0) in vec4 position;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 passDir;

void main() 
{
    gl_Position = position;

    mat4 inverseView = inverse(viewMatrix);
    mat4 inverseProjection = inverse(projectionMatrix);

    vec4 origin = inverseView * vec4(0,0,0,1);
    vec4 originPlusDir = (inverseView * inverseProjection) * vec4(position.xy, 0, 1);
    originPlusDir.xyz /= originPlusDir.w;
    passDir = normalize(vec3(originPlusDir - origin));
}