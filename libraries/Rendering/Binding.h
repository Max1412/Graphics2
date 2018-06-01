#pragma once

#include "glsp/glsp.hpp"

namespace BufferBindings
{
    enum class Binding : int
    {
        cameraParameters = 7,
        lights = 8,
        materials = 9,
        modelMatrices = 10,
        materialIndices = 11
    };

    enum class VertexAttributeLocation : int
    {
        vertices = 0,
        normals = 1,
        texCoords = 2
    };

    enum class Subroutine : int
    {
        multiDraw = 0,
        normalDraw = 1
    };

    inline std::vector<glsp::definition> g_definitions = {
        glsp::definition("CAMERA_BINDING", static_cast<int>(Binding::cameraParameters)),
        glsp::definition("LIGHTS_BINDING", static_cast<int>(Binding::lights)),
        glsp::definition("MATERIAL_BINDING", static_cast<int>(Binding::materials)),
        glsp::definition("MODELMATRICES_BINDING", static_cast<int>(Binding::modelMatrices)),
        glsp::definition("MATERIAL_INDICES_BINDING", static_cast<int>(Binding::materialIndices)),


        glsp::definition("VERTEX_LAYOUT", static_cast<int>(VertexAttributeLocation::vertices)),
        glsp::definition("NORMAL_LAYOUT", static_cast<int>(VertexAttributeLocation::normals)),
        glsp::definition("TEXCOORD_LAYOUT", static_cast<int>(VertexAttributeLocation::texCoords))

    };
}

