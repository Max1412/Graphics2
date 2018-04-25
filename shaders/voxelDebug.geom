#version 430

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 posColor[];
out vec3 geomColor;

uniform mat4 dViewMat;
uniform mat4 dProjMat;

void main() {
    vec4 position = gl_in[0].gl_Position;
    geomColor = posColor[0];
    vec4 pos = vec4(1.0f);

    pos = position + vec4(-0.2, -0.2, 0.0, 0.0);    // 1:bottom-left
    gl_Position = dProjMat * dViewMat * pos;
    EmitVertex();   

    pos = position + vec4( 0.2, -0.2, 0.0, 0.0);    // 2:bottom-right
    gl_Position = dProjMat * dViewMat * pos;
    EmitVertex();

    pos = position + vec4(-0.2,  0.2, 0.0, 0.0);    // 3:top-left
    gl_Position = dProjMat * dViewMat * pos;
    EmitVertex();

    pos = position + vec4( 0.2,  0.2, 0.0, 0.0);    // 4:top-right
    gl_Position = dProjMat * dViewMat * pos;
    EmitVertex();
    
    EndPrimitive();
}  

