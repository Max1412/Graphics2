#version 430
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

uniform mat4 lightSpaceMatrix;
uniform vec3 lightPos;

mat4 shadowMatrices[6] = { 
    mat4(vec4(0, 0, -1, 0), vec4(0, -1, 0, 0), vec4(-1, 0, 0, 0), vec4(0, 0, 0, 1)),
    mat4(vec4(0, 0,  1, 0), vec4(0, -1, 0, 0), vec4( 1, 0, 0, 0), vec4(0, 0, 0, 1)),
    mat4(vec4( 1, 0, 0, 0), vec4(0, 0, -1, 0), vec4(0,  1, 0, 0), vec4(0, 0, 0, 1)),
    mat4(vec4( 1, 0, 0, 0), vec4(0, 0,  1, 0), vec4(0, -1, 0, 0), vec4(0, 0, 0, 1)),
    mat4(vec4( 1, 0, 0, 0), vec4(0, -1, 0, 0), vec4(0, 0, -1, 0), vec4(0, 0, 0, 1)),
    mat4(vec4(-1, 0, 0, 0), vec4(0, -1, 0, 0), vec4(0, 0,  1, 0), vec4(0, 0, 0, 1))
};

out vec4 FragPos; // FragPos from GS (output per emitvertex)

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            mat4 view = shadowMatrices[face];
            view[3] = vec4(lightPos, 1);
            gl_Position = lightSpaceMatrix * view * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
}  