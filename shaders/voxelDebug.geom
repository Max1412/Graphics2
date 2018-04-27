#version 430

layout (points) in;
layout (triangle_strip, max_vertices = 24) out;

in vec3 posColor[];
out vec3 geomColor;

uniform mat4 dViewMat;
uniform mat4 dProjMat;

float size = 0.5f;

void main() {
    vec4 position = gl_in[0].gl_Position;
    geomColor = posColor[0];
    vec4 pos = vec4(1.0f);
    mat4 pvMat = dProjMat * dViewMat;

    vec4[8] offsets = { vec4(-size, -size, size, 0.f),
                        vec4(size, -size, size, 0.f),
                        vec4(-size, size, size, 0.f),
                        vec4(size, size, size, 0.f),
                        vec4(-size, -size, -size, 0.f),
                        vec4(size, -size, -size, 0.f),
                        vec4(-size, size, -size, 0.f),
                        vec4(size, size, -size, 0.f)};

    { // front
        pos = position + offsets[0];    // 1:bottom-left
        gl_Position = pvMat * pos;
        EmitVertex();   

        pos = position + offsets[1];    // 2:bottom-right
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[2];    // 3:top-left
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[3];    // 4:top-right
        gl_Position = pvMat * pos;
        EmitVertex();
    
        EndPrimitive();
    }

    { // back
        pos = position + offsets[5];    // 1:bottom-left
        gl_Position = pvMat * pos;
        EmitVertex();   

        pos = position + offsets[4];    // 2:bottom-right
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[7];    // 3:top-left
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[6];    // 4:top-right
        gl_Position = pvMat * pos;
        EmitVertex();
    
        EndPrimitive();
    }

    { // left
        pos = position + offsets[4];    // 1:bottom-left
        gl_Position = pvMat * pos;
        EmitVertex();   

        pos = position + offsets[0];    // 2:bottom-right
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[6];    // 3:top-left
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[2];    // 4:top-right
        gl_Position = pvMat * pos;
        EmitVertex();
    
        EndPrimitive();
    }

    { // right
        pos = position + offsets[1];    // 1:bottom-left
        gl_Position = pvMat * pos;
        EmitVertex();   

        pos = position + offsets[5];    // 2:bottom-right
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[3];    // 3:top-left
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[7];    // 4:top-right
        gl_Position = pvMat * pos;
        EmitVertex();
    
        EndPrimitive();
    }

    { // top
        pos = position + offsets[2];    // 1:bottom-left
        gl_Position = pvMat * pos;
        EmitVertex();   

        pos = position + offsets[3];    // 2:bottom-right
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[6];    // 3:top-left
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[7];    // 4:top-right
        gl_Position = pvMat * pos;
        EmitVertex();
    
        EndPrimitive();
    }

    { // bottom
        pos = position + offsets[0];    // 1:bottom-left
        gl_Position = pvMat * pos;
        EmitVertex();   

        pos = position + offsets[4];    // 2:bottom-right
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[1];    // 3:top-left
        gl_Position = pvMat * pos;
        EmitVertex();

        pos = position + offsets[5];    // 4:top-right
        gl_Position = pvMat * pos;
        EmitVertex();
    
        EndPrimitive();
    }
}  

