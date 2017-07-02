#version 430

in vec3 TexCoord;

layout (location = 0) out vec4 FragColor;

layout (std430, binding = 0) restrict readonly buffer BlobSettings {
	vec4 InnerColor;
	vec4 OuterColor;
	float RadiusInner;
	float RadiusOuter;
};

void main() {
	float dx = TexCoord.x - 0.5;
	float dy = TexCoord.y - 0.5;
	float dist = sqrt(dx * dx + dy * dy);
	FragColor = mix( InnerColor, OuterColor, smoothstep( RadiusInner, RadiusOuter, dist));
}