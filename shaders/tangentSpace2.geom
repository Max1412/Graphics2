#version 430

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 passFragPos[];
layout(location = 1) in vec3 passTexCoord[];
layout(location = 2) in vec3 passNormal[];
layout(location = 3) in vec3 passViewPos[];
layout(location = 4) flat in uint passDrawID[];

layout(location = 0) out vec3 pos;
layout(location = 1) out vec3 texCoord;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 viewPos;
layout(location = 4) flat out uint drawID;
layout(location = 5) out vec3 tangent;
layout(location = 6) out vec3 bitangent;

void main() {

    vec3 edge1 = passFragPos[1] - passFragPos[0];
	vec3 edge2 = passFragPos[2] - passFragPos[0];
	vec2 deltaUV1 = passTexCoord[1].xy - passTexCoord[0].xy;
	vec2 deltaUV2 = passTexCoord[2].xy - passTexCoord[0].xy; 

	float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

	tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
	tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
	tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
	tangent = normalize(tangent);

	bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
	bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
	bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
	bitangent = normalize(bitangent);

	drawID = passDrawID[0];

	for (int i = 0; i < 3; i++) {
		gl_Position = gl_in[i].gl_Position;
        texCoord = passTexCoord[i];
        pos = passFragPos[i];
		normal = normalize(passNormal[i]);
		viewPos = passViewPos[i];
        EmitVertex();
	}
	EndPrimitive();
}  

