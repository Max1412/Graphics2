#pragma once

#include "Mesh.h"

class SkyBoxCube : public Mesh
{
public:
    SkyBoxCube();
private:
    static std::vector<glm::vec3> cubeVertices;
    static std::vector<unsigned> cubeIndices;
    static std::vector<glm::vec3> noNormals;

};