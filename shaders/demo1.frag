#version 430

in vec3 Position;
in vec3 interpNormal;
flat in vec3 flatNormal;

layout (std430, binding = 0) restrict readonly buffer Light {
	vec4 LightPosition; // Light position in eye coords.
	vec3 LightIntensity;
};

layout (std430, binding = 1) restrict readonly buffer Material {
	vec3 Ka; // Ambient reflectivity
	vec3 Kd; // Diffuse reflectivity
	vec3 Ks; // Specular reflectivity
	float Shininess; // Specular shininess factor
};

uniform int useFlat;

layout( location = 0 ) out vec4 FragColor;

vec3 ads( ) {
	vec3 Normal = vec3(0.0, 0.0, 0.0);
	if(useFlat == 1){
		Normal = flatNormal;
	} else {
		Normal = interpNormal;
	}
	vec3 n = normalize( Normal );
	vec3 s = normalize( vec3(LightPosition) - Position );
	vec3 v = normalize(vec3(-Position));
	vec3 r = reflect( -s, n );
	return LightIntensity *	( Ka + Kd * max( dot(s, n), 0.0 ) +	Ks * pow( max( dot(r,v), 0.0 ), Shininess ) );
}

void main() {
	FragColor = vec4(ads(), 1.0);
}