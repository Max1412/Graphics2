#pragma once

#include <glbinding/gl/gl.h>
using namespace gl;

#include <memory>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

#include "Mesh.h"
#include "ShaderProgram.h"

class SparseVoxelOctree 
{
public:
    SparseVoxelOctree(const std::vector<std::shared_ptr<Mesh>>& scene, const size_t depth);

    void bind() const;

    void update();

    glm::vec3 getBMin() const;
    glm::vec3 getBMax() const;

protected:

    ShaderProgram m_voxelGenShader{ {
        Shader{ "SparseVoxelOctree/VoxelGen.vert", GL_VERTEX_SHADER},
        Shader{ "SparseVoxelOctree/VoxelGen.geom", GL_GEOMETRY_SHADER},
        Shader{ "SparseVoxelOctree/VoxelGen.frag", GL_FRAGMENT_SHADER}
    } };

    ShaderProgram m_flagShader{ {
        Shader{ "SparseVoxelOctree/FlagNodes.comp", GL_COMPUTE_SHADER }
    } };

    ShaderProgram m_nodeCreationShader{ {
        Shader{ "SparseVoxelOctree/CreateNodes.comp", GL_COMPUTE_SHADER }
    } };

    ShaderProgram m_leafInitShader{ {
        Shader{ "SparseVoxelOctree/InitLeafNodes.comp", GL_COMPUTE_SHADER }
    } };

    ShaderProgram m_mipMapShader{ {
        Shader{ "SparseVoxelOctree/MipMapNodes.comp", GL_COMPUTE_SHADER }
    } };

    Buffer m_voxelFragmentList{GL_SHADER_STORAGE_BUFFER};   // vec4
    Buffer m_voxelCounter{GL_ATOMIC_COUNTER_BUFFER};        // uint32
    Buffer m_voxelFragmentColor{GL_SHADER_STORAGE_BUFFER};  // vec4

    Buffer m_nodePool{GL_SHADER_STORAGE_BUFFER};        // int32
    Buffer m_nodeCounter{GL_ATOMIC_COUNTER_BUFFER};     // uint32
    Buffer m_nodeColor{GL_SHADER_STORAGE_BUFFER};       // vec4

    std::shared_ptr<Uniform<int>> m_startIndexUniform;
    std::shared_ptr<Uniform<glm::mat4>> m_modelMatrixUniform;

    const size_t m_N;
    const size_t m_depth;
    const std::vector<std::shared_ptr<Mesh>>& m_scene;

    glm::vec3 m_bmin;
    glm::vec3 m_bmax;
};
