#version 430

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;

out vec3 passPosition;
out vec3 interpNormal;


uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

void main()
{
	mat4 ModelViewMatrix = ViewMatrix * ModelMatrix;
	mat3 NormalMatrix = mat3(transpose(inverse(ModelViewMatrix)));

	vec3 Normal = normalize( NormalMatrix * VertexNormal);
	interpNormal = Normal;
	passPosition = vec3( ModelViewMatrix * 	vec4(VertexPosition, 1.0));
	gl_Position = ProjectionMatrix * ModelViewMatrix * vec4(VertexPosition, 1.0);
}