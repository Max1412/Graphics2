#version 430 core
out vec4 fragColor;
in vec3 geomColor;

void main()
{
    fragColor = vec4(geomColor, 1.0);   
}  