#version 430

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;

flat out vec3 LightIntensity;

layout (std430, binding = 0) restrict readonly buffer Light {
	vec4 Position; // Light position in eye coords.
	vec3 La; // Ambient light intensity
	vec3 Ld; // Diffuse light intensity
	vec3 Ls; // Specular light intensity
};

layout (std430, binding = 1) restrict readonly buffer Material {
	vec3 Ka; // Ambient reflectivity
	vec3 Kd; // Diffuse reflectivity
	vec3 Ks; // Specular reflectivity
	float Shininess; // Specular shininess factor
};

uniform mat4 ViewMatrix;
uniform mat4 ModelMatrix;
uniform mat4 ProjectionMatrix;

void main() {
	mat3 NormalMatrix = mat3(transpose(inverse(ViewMatrix * ModelMatrix)));
	mat4 ModelViewMatrix = ViewMatrix * ModelMatrix;

	vec3 tnorm = normalize( NormalMatrix * VertexNormal);
	vec4 eyeCoords = ModelViewMatrix * vec4(VertexPosition,1.0);

	vec3 s = normalize(vec3(Position - eyeCoords));
	vec3 v = normalize(-eyeCoords.xyz);
	vec3 r = reflect( -s, tnorm );

	vec3 ambient = La * Ka;
	float sDotN = max( dot(s,tnorm), 0.0000001f);
	vec3 diffuse = Ld * Kd * sDotN;
	vec3 spec = vec3(0.0);

	if( sDotN > 0.0 )
		spec = Ls * Ks * pow(max( dot(r,v), 0.0000001f ), Shininess);

	LightIntensity = ambient + diffuse + spec;

	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(VertexPosition,1.0);
}