#version 430

in vec3 Position;
in vec3 interpNormal;
flat in vec3 flatNormal;

struct Light {
	vec4 LightPosition; // Light position in eye coords.
	vec3 LightIntensity;
};

struct Material {
	vec3 Ka; // Ambient reflectivity
	vec3 Kd; // Diffuse reflectivity
	vec3 Ks; // Specular reflectivity
	float Shininess; // Specular shininess factor
};

layout (std430, binding = 0) restrict readonly buffer LightBuffer {
	Light Lights[];
};

layout (std430, binding = 1) restrict readonly buffer MaterialBuffer {
	Material Materials[];
};


uniform int useFlat;
uniform int useToon;
uniform int mID;

layout( location = 0 ) out vec4 FragColor;

vec3 ads(int lightIndex, vec3 pos, vec3 norm) {
	vec3 v = normalize(-pos);
	vec3 s = normalize( vec3(Lights[lightIndex].LightPosition) - pos );
	vec3 r = reflect( -s, norm );
	return Lights[lightIndex].LightIntensity *
	( Materials[mID].Ka + Materials[mID].Kd * max( dot(s, norm), 0.0000001 ) +
	Materials[mID].Ks * pow( max( dot(r,v), 0.000001 ), Materials[mID].Shininess ) );
}

uniform int levels;

vec3 toonShade(int lightIndex, vec3 pos, vec3 norm) {
	float scaleFactor = 1.0 / levels;
	vec3 s = normalize( vec3(Lights[lightIndex].LightPosition) - pos );
	float cosine = max( 0.000001, dot( s, norm ) );
	vec3 diffuse = Materials[mID].Kd * floor( cosine * levels ) * scaleFactor;
	return Lights[lightIndex].LightIntensity * (Materials[mID].Ka + diffuse);
}

void main() {
	vec3 Normal = vec3(0.0, 0.0, 0.0);
	if(useFlat == 1){
		Normal = flatNormal;
	} else {
		Normal = interpNormal;
	}
	vec3 n = normalize( Normal );
	FragColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	for(int i = 0; i < Lights.length(); i++){
		if(useToon == 1){
			FragColor += vec4(toonShade(i, Position, n ), 1.0);
		} else {
			FragColor += vec4(ads(i, Position, n ), 1.0);
		}
	}
}