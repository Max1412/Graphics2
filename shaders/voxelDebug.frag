#version 430 core
out vec4 fragColor;
in vec3 geomColor;

uniform mat4 dProjMat;
uniform mat4 dViewMat;

void main()
{
    fragColor = vec4(geomColor, 1.0);   
}  