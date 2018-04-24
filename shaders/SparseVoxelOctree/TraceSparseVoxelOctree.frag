#version 450

in vec3 passDir;
uniform vec3 camPosition;

layout(binding = 2, std430) buffer nodePool_buffer
{
    int nodePool[];
};

layout(binding = 3, std430) buffer nodeColor_buffer
{
    vec4 nodeColor[];
};

#define is_leaf (node == -1)

void handleLeaf(in int nodeID, inout vec4 color, inout bool hit)
{
    if(nodeColor[nodeID].w > 0.0f)
    {
        hit = true;
        color = nodeColor[nodeID];
    }
}

float compMax(vec3 v)
{
    return max(v.x,max(v.y,v.z));
}

float compMin(vec3 v)
{
    return min(v.x,min(v.y,v.z));
}

bool rayBoxIntersect( vec3 rpos, vec3 rdir, vec3 vmin, vec3 vmax, out float tMin, out float tMax)
{
   vec3 t1 = (vmin - rpos)/rdir;
   vec3 t2 = (vmax - rpos)/rdir;
   tMin = compMax(min(t1,t2));
   tMax = compMin(max(t1,t2));
   return tMin < tMax;
}

bool rayInvBoxIntersect( vec3 rpos, vec3 rdir, vec3 vmin, vec3 vmax, out float tMin, out float tMax)
{
   vec3 t1 = (vmin - rpos)*rdir;
   vec3 t2 = (vmax - rpos)*rdir;
   tMin = compMax(min(t1,t2));
   tMax = compMin(max(t1,t2));
   return tMin < tMax;
}

uniform int maxLevel = 10; //only for level debugging!
uniform vec3 bmin;
uniform vec3 bmax;

//stackless while if
vec4 trace(vec3 origin, vec3 dir, float tMin, float tMax)
{
    if(!rayBoxIntersect(origin,dir,bmin,bmax,tMin,tMax) || tMax < 0.f)
    {
        return vec4(-1);
    }
    tMin = max(tMin,0.f);

    vec3 rootSize = bmax - bmin;
    ivec3 nodePosition = ivec3(0,0,0);
    ivec3 cIndex = ivec3(greaterThanEqual((origin + tMin * dir - bmin), vec3(0.5f * rootSize))); //clamp(ivec3(2.0f * (origin + tMin * dir - bmin) / rootSize),0,1);

    int nodeID = 0;
    int stackPointer = 0;
    int voxelSizeFactor = 1;

    bool hit = false;

    dir = max(abs(dir),vec3(1e-7)) * sign(dir);
    vec3 invDir = 1.f/dir;
    vec3 step = vec3(dir.x < 0.f ? -1.f : 1.f, dir.y < 0.f ? -1.f : 1.f, dir.z < 0.f ? -1.f : 1.f);
    int node;

    int stack[10/*MAX_DEPTH*/];
    stack[0] = 0;

     while(stackPointer >= 0)
     {
          node = nodePool[nodeID];
          //check if current node is not leaf node
          if(node > 0 && cIndex.x != -1)
          {
              stackPointer++;
              nodeID = node + cIndex.x + (cIndex.y << 1 /* *2 */) + (cIndex.z << 2 /* *2*2 */);
              stack[stackPointer] = nodeID;
              nodePosition = (nodePosition << 1 /* *2 */) + cIndex;
              voxelSizeFactor <<= 1; // *2

              //only for level debugging!
              if(stackPointer >= maxLevel)
              {
                    vec4 tr;
                    handleLeaf(nodeID,tr,hit);
                    if(hit)
                    {
                        return tr;
                    }
              }

              vec3 voxelSize = rootSize / voxelSizeFactor;
              vec3 minPos = vec3(nodePosition) * voxelSize + bmin;
              vec3 maxPos = minPos + voxelSize;
              rayInvBoxIntersect(origin,invDir,minPos,maxPos,tMin,tMax);
              cIndex = ivec3(greaterThanEqual((origin + max(0.f,tMin) * dir - minPos), vec3(0.5f * voxelSize))); //clamp(ivec3(2.0f * (origin + max(0.f,tMin) * dir - minPos) / voxelSize),0,1);

          }
          else
          {
               if(node == -1)
               {
                    vec4 tr;
                    handleLeaf(nodeID,tr,hit);
                    if(hit)
                    {
                        return tr;
                    }
                }

              //this node is not needed on the stack anymore
              stackPointer--;
              nodeID = stack[stackPointer];
              cIndex = nodePosition & 0x1; // %2
              nodePosition >>= 1; // /2
              voxelSizeFactor >>= 1; // /2

              //calculate next childIndex.
              vec3 voxelSize = rootSize / voxelSizeFactor;
              vec3 position = (vec3(nodePosition) + vec3(cIndex)*0.5f + 0.25f + step * 0.25f) * voxelSize + bmin;
              vec3 tStep = (position - origin) * invDir;
              float tMin2 = compMin(tStep);
              int tStepMinIndex = int(tStep.y == tMin2) + (int(tStep.z == tMin2) << 1 /* *2 */);
              int cIndexNew = cIndex[tStepMinIndex] += int(step[tStepMinIndex]);
              if(cIndexNew < 0 || cIndexNew > 1){
                  cIndex.x = -1;
              }
          }
     }
    return vec4(-1);
}

void main()
{
    ivec2 pixelPos = ivec2(gl_FragCoord.xy);
    vec3 dir = normalize(passDir);
    gl_FragColor = vec4(vec3(0.7f, 0.9f, 1.0f) + dir.y * 0.618f, 1.0f);

    vec4 r = trace(camPosition,dir,0,1000);
    if(r.w >= 0)
    {
        gl_FragColor = r;
    }
}