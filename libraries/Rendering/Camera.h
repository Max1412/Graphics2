#pragma once

#include <glbinding/gl/gl.h>
using namespace gl;
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

struct FrustumGeo
{
    enum Plane
    {
        TOP = 0, BOTTOM = 1, LEFT = 2,
        RIGHT = 3, NEAR = 4, FAR = 5
    };

    std::array<glm::vec3, 6> normals;
    std::array<glm::vec3, 8> points;

    // the function setNormalFromPoints assumes that the points
    // are given in counter clockwise order
    void setNormalFromPoints(Plane plane, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
    {
        normals[plane] = glm::normalize(glm::cross(v3 - v2, v1 - v2));
    }

    // gets the smallest distance from a point to a plane
    float distance(int plane, const glm::vec3& p) 
    {
        return glm::dot(normals[plane], p - points[plane]);
    }
};

class Camera
{
public:
    Camera(int width, int height);

    /**
     * \brief Updates the view matrix based on mouse input
     * \param window 
     */
    virtual void update(GLFWwindow* window) = 0;

    virtual void reset();

    /**
     * \brief returns the view matrix (const)
     * \return 4x4 view matrix (const)
     */
    const glm::mat4& getView() const;

    /**
     * \brief returns the view matrix (mutable)
     * \return 4x4 view matrix
     */
    glm::mat4& getView();

    glm::vec3 getPosition();
    glm::vec3 getCenter();
    glm::vec3 getDirection();

    void setPosition(glm::vec3 pos);

    virtual ~Camera() = default;

protected:
    glm::mat4 m_viewMatrix;
    glm::vec3 m_center;
    glm::vec3 m_pos;
    glm::vec3 m_up;

    int m_width;
    int m_height;

    float m_sensitivity;
    float m_theta;
    float m_phi;
    float m_oldX;
    float m_oldY;
};
