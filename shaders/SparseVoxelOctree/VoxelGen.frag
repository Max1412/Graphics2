#version 450 core
//#extension GL_NV_shader_atomic_float : enable
//const can slow down drastically
#define const

//enable 3D conservative rasterization
#define WATERTIGHT

//layout(pixel_center_integer) in vec4 gl_FragCoord;

uniform uvec3 res;

layout(binding = 0, std430) buffer voxelFragmentList_buffer
{
    vec4 voxelFragmentList[];
};

layout(binding = 1, std430) buffer voxelFragmenColor_buffer
{
    vec4 voxelFragmentColor[];
};

layout (binding = 0, offset = 0) uniform atomic_uint voxelCounter;

uniform struct 
{
    float kd, ks, kt;
    vec3 diffColor;
    vec3 specColor;
    float shininess;
} mat;

flat in int axis;
flat in vec3 face_normal;
flat in vec3 vertex0, vertex1, vertex2;
flat in vec3 diag;

//out vec4 fragmentColor;

vec2 dmin;

void compute_critical_point_directions(const in  vec3 normal, const in vec3 d)
{
    dmin.x = (normal.x > 0.0f) ? -d.x : d.x;
    dmin.y = (normal.y > 0.0f) ? -d.y : d.y;
}

bool triBoxOverlap( const in vec3 voxCenter, const in vec3 d)
{
    vec2 minMaxValue;
    float p0, p1, r;
    vec3 v0 = vertex0 - voxCenter;
    vec3 v1 = vertex1 - voxCenter;
    vec3 v2 = vertex2 - voxCenter;

    vec3 e0 = v1 - v0;
    vec3 e0_abs = abs(e0);    

    // AXISTEST_x_e0
    p0 = e0.z * v0.y - e0.y * v0.z;
    p1 = e0.z * v2.y - e0.y * v2.z;
    minMaxValue = (p0 < p1) ? vec2(p0,p1) : vec2(p1,p0);

    r = e0_abs.z * d.y + e0_abs.y * d.z;
    if (minMaxValue.x > r || minMaxValue.y < -r) return false;

    // AXISTEST_y_e0
    p0 = -e0.z * v0.x + e0.x * v0.z;
    p1 = -e0.z * v2.x + e0.x * v2.z;
    minMaxValue = (p0 < p1) ? vec2(p0,p1) : vec2(p1,p0);

    r = e0_abs.z * d.x + e0_abs.x * d.z;
    if (minMaxValue.x > r || minMaxValue.y < -r) return false;
    
    vec3 e1 = v2 - v1;
    vec3 e1_abs = abs(e1);

    // AXISTEST_x_e1
    p0 = e1.z * v0.y - e1.y * v0.z;
    p1 = e1.z * v2.y - e1.y * v2.z;
    minMaxValue = (p0 < p1) ? vec2(p0,p1) : vec2(p1,p0);

    r = e1_abs.z * d.y + e1_abs.y * d.z;
    if (minMaxValue.x > r || minMaxValue.y < -r) return false;

    // AXISTEST_y_e1
    p0 = -e1.z * v0.x + e1.x * v0.z;
    p1 = -e1.z * v2.x + e1.x * v2.z;
    minMaxValue = (p0 < p1) ? vec2(p0,p1) : vec2(p1,p0);

    r = e1_abs.z * d.x + e1_abs.x * d.z;
    if (minMaxValue.x > r || minMaxValue.y < -r) return false;
    
    vec3 e2 = v0 - v2;
    vec3 e2_abs = abs(e2);

    // AXISTEST_x_e2
    p0 = e2.z * v0.y - e2.y * v0.z;
    p1 = e2.z * v1.y - e2.y * v1.z;
    minMaxValue = (p0 < p1) ? vec2(p0,p1) : vec2(p1,p0);

    r = e2_abs.z * d.y + e2_abs.y * d.z;
    if (minMaxValue.x > r || minMaxValue.y < -r) return false;

    // AXISTEST_y_e2
    p0 = -e2.z * v0.x + e2.x * v0.z;
    p1 = -e2.z * v1.x + e2.x * v1.z;
    minMaxValue = (p0 < p1) ? vec2(p0,p1) : vec2(p1,p0);

    r = e2_abs.z * d.x + e2_abs.x * d.z;
    if (minMaxValue.x > r || minMaxValue.y < -r) return false;

    return true;
}

int minIvec3(in ivec3 vec)
{
    return min(min(vec.x,vec.y),vec.z);
}
int maxIvec3(in ivec3 vec)
{
    return max(max(vec.x,vec.y),vec.z);
}

uint encodeVoxelPos(uvec3 tc) // max value per channel is 1023!
{
    return 0x80000000 | (tc.z & 0x000003FF) << 20U | (tc.y & 0x000003FF) << 10U | (tc.x & 0x000003FF);
}
uvec4 decodeVoxelPos(uint tc) // max value per channel is 1023!
{
    return uvec4(tc & 0x000003FF, (tc & 0x000FFC00) >> 10U, (tc & 0x3FF00000) >> 20U, (tc & 0x80000000) >> 31U);
}

void main()
{
    ivec3 tc;

    #ifdef WATERTIGHT
    // compute intersections for critical points and snap to grid
    compute_critical_point_directions(face_normal, diag);

    vec2 c =  vec2((2.0f*gl_FragCoord.x)*diag.x - 1.f, (2.0f*gl_FragCoord.y)*diag.y - 1.f);
    float dist = -dot(face_normal, vertex0);

    float zmin = (-dist - dot( c + dmin, face_normal.xy)) / face_normal.z;
    float zmax = (-dist - dot( c - dmin, face_normal.xy)) / face_normal.z;

    int res_z = int(res[axis-1]);
    //res_z = int(1 / diag.z); // !what a bug! int(1.f/1.f/120.f) = 119

    int z_index_cmin = (zmax >= 1.0) ? 0 : int(res_z - res_z * (zmax + 1.f) / 2.f);
    int z_index_cmax = (zmin <= -1.0f) ? res_z - 1 : int(res_z - res_z * (zmin + 1.f) / 2.f);

    //voxel index as an integer: from 0 to res-1
    if(min(z_index_cmin,z_index_cmax) < 0 || max(z_index_cmin,z_index_cmax) >= res_z) return;

    for ( int z = min(z_index_cmin,z_index_cmax); z <= max(z_index_cmin,z_index_cmax); ++z)
        if (triBoxOverlap(vec3(c, -2.f*(z+0.5f)*diag.z+1.f), diag))
        {
            if (axis == 1)
                tc  = ivec3 (res.x - z - 1, gl_FragCoord.x, res.z - gl_FragCoord.y);
            else if (axis == 2)
                tc  = ivec3 (gl_FragCoord.y, res.y - z - 1, res.z - gl_FragCoord.x);
            else
                tc  = ivec3 (gl_FragCoord.x, gl_FragCoord.y, z);

    #else

        {    
            if (axis == 1)
                tc  = ivec3 (res.x * gl_FragCoord.z, gl_FragCoord.x, res.z - gl_FragCoord.y);
            else if (axis == 2)
                tc  = ivec3 (gl_FragCoord.y, res.y * gl_FragCoord.z, res.z - gl_FragCoord.x);
            else
                tc  = ivec3 (gl_FragCoord.x, gl_FragCoord.y, res.z - res.z * gl_FragCoord.z);

            //voxel index as an integer: from 0 to res-1
            if(minIvec3(tc) < 0 || any(greaterThanEqual(tc, res))) return;

    #endif
            

            int index = int(atomicCounterIncrement(voxelCounter));
            //index = uint(tc.z * res.y* res.x + tc.y * res.x + tc.x); //only for testing!
            voxelFragmentList[index] = vec4(tc/vec3(res),1); 

            vec3 color = tc/vec3(res); //TODO put color in here!
            voxelFragmentColor[index] = vec4(color, 1.0f);

        }

    //fragmentColor = vec4(1.f);
}
