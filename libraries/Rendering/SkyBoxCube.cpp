#include "SkyBoxCube.h"

SkyBoxCube::SkyBoxCube() : Mesh(cubeVertices, noNormals, cubeIndices)
{
    // this constructor initlializes the Mesh with the pre-defined vertices and indices, and with empty normals
}

std::vector<glm::vec3> SkyBoxCube::cubeVertices = {
    // front
    { -1.0, -1.0, 1.0 },
    { 1.0, -1.0, 1.0 },
    { 1.0, 1.0, 1.0 },
    { -1.0, 1.0, 1.0 },
    // back
    { -1.0, -1.0, -1.0 },
    { 1.0, -1.0, -1.0 },
    { 1.0, 1.0, -1.0 },
    { -1.0, 1.0, -1.0 },
};

std::vector<unsigned> SkyBoxCube::cubeIndices = {
    // front
    0, 1, 2,
    2, 3, 0,
    // top
    1, 5, 6,
    6, 2, 1,
    // back
    7, 6, 5,
    5, 4, 7,
    // bottom
    4, 0, 3,
    3, 7, 4,
    // left
    4, 5, 1,
    1, 0, 4,
    // right
    3, 2, 6,
    6, 7, 3,
};

std::vector<glm::vec3> SkyBoxCube::noNormals;