#version 450 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 projectionMatrix;
uniform uvec3 res;

flat out vec3 vertex0, vertex1, vertex2;

flat out int axis;
flat out vec3 face_normal;
flat out vec3 diag;

void main(void)
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

    vec4 pos[3] = 
    {
        projectionMatrix * gl_in[0].gl_Position,
        projectionMatrix * gl_in[1].gl_Position,
        projectionMatrix * gl_in[2].gl_Position
    };

    face_normal = normalize(cross((pos[2] - pos[1]).xyz, (pos[1] - pos[0]).xyz));
    vec3 n = abs(face_normal);
    diag = vec3(res.xyz);
    axis = 3;

    //Find the axis that maximize the projected area of this triangle
    if(n.x > n.y && n.x > n.z)
    {
        pos[0] = pos[0].yzxw;
        pos[1] = pos[1].yzxw;
        pos[2] = pos[2].yzxw;
        diag = diag.yzx;
        face_normal = face_normal.yzx;
        axis = 1;
    }
    else if(n.y > n.x && n.y > n.z)
    {
        pos[0] = pos[0].zxyw;
        pos[1] = pos[1].zxyw;
        pos[2] = pos[2].zxyw;
        diag = diag.zxy;
        face_normal = face_normal.zxy;
        axis = 2;
    }

    diag = 1.0f / diag;

    vertex0 = pos[0].xyz;
    vertex1 = pos[1].xyz;
    vertex2 = pos[2].xyz;

    gl_ViewportIndex = axis;

    gl_Position = pos[0];
    EmitVertex();

    gl_Position = pos[1];
    EmitVertex();

    gl_Position = pos[2];
    EmitVertex();

    EndPrimitive();
}
