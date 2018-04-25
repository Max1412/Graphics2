#version 430
#extension GL_ARB_bindless_texture : require
out vec3 voxelCenterPosition;

uniform mat4 debugViewMat;
uniform mat4 debugProjMat;
uniform ivec3 gridDim;

layout(binding = 0, std430) buffer voxelGridBuffer
{
	layout(rgba32f) image3D voxelGrid;
};

void main()
{
    int id1D = gl_VertexID;
    ivec3 gridPos3D = ivec3(mod(id1D, gridDim.x), mod(id1D / gridDim.x, gridDim.y), id1D / (gridDim.x * gridDim.y));
    vec4 voxelDataContent = imageLoad(voxelGrid, gridPos3D);
    voxelCenterPosition = (debugProjMat * debugViewMat * voxelDataContent).xyz;
}