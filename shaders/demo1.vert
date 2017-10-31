#version 430

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;

out vec3 passPosition;
out vec3 interpNormal;
flat out vec3 flatNormal;
out vec3 modelNormal;
out vec3 Position;


uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

void main()
{
	mat4 ModelViewMatrix = ViewMatrix * ModelMatrix;
	mat3 NormalMatrix = mat3(transpose(inverse(ModelViewMatrix)));

	vec3 Normal = normalize( NormalMatrix * VertexNormal);
	interpNormal = Normal;
	flatNormal = Normal;
	passPosition = vec3( ModelViewMatrix * 	vec4(VertexPosition, 1.0));

    modelNormal = mat3(transpose(inverse(ModelMatrix))) * VertexNormal;
    Position = vec3(ModelMatrix * vec4(VertexPosition, 1.0));

	gl_Position = ProjectionMatrix * ModelViewMatrix * vec4(VertexPosition, 1.0);
}