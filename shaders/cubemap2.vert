#version 430
layout (location = 0) in vec3 vertexPosition;

out vec3 texCoords;

layout(binding = CAMERA_BINDING, std430) buffer cameraBuffer
{
    mat4 ViewMatrix;
    mat4 ProjectionMatrix;
    vec3 camPos;
};

void main()
{
    texCoords = vertexPosition;
    vec4 pos = ProjectionMatrix * mat4(mat3(ViewMatrix)) * vec4(vertexPosition, 1.0f);
    gl_Position = pos.xyww;
}