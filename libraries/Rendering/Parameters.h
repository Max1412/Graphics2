#pragma once

#include <vector>
#include <glm/glm.hpp>

struct ParameterFog
{
	glm::vec3 albedo;
	float anisotropy;
	float scattering;
	float absorption;
	float density;
};

struct PrarameterLight
{
	glm::vec3 color;        // all
	glm::vec3 position;     // spot, point
	glm::vec3 direction;    // dir, spot
	int pcfKernelSize = 1;
	float constant;         // spot, point
	float linear;           // spot, point
	float quadratic;        // spot, point
	float cutOff;           // spot
	float outerCutOff;      // spot
};

struct ParamterNoise
{
	float densityFactor;
	float densityHeight;
	float scale;
	float speed;
};

struct Parameters
{
	std::vector<PrarameterLight> lights;
	ParamterNoise noise;
	ParameterFog fog;
	float gamma;
	float exposure;
	float maxRange; //range of voxel grid
	glm::vec3 cameraPos;
	float phi, theta; //for camera direction

};

/* E X A M P L E
Parameters Scene{
  LIGHTS
	{
		{color, position, direction, pcfKernelSize, constant, linear, quadratic, cutOff, outerCutoff},
		...
	},
  NOISE
	{densityFactor, densityHeigt, scale, speed},
  FOG
	{albedo, anisotropy, scattering, absorption, density},
  HDR AND VOXEL SETTINGS
	gamma, exposure, maxRange,
  CAMERA PARAMETERS
    cameraPos, phi, theta
}*/

const std::vector<Parameters> sceneParams = {

    // S P O N Z A
Parameters{
    {
        //global directional light
        {glm::vec3(10.0f), glm::vec3(0.0f, 2000.0f, 0.0f), glm::vec3(0.0f, -1.0f, -0.2f)},
        //spotlight 1
        {glm::vec3(0.0f, 10.0f, 10.0f), glm::vec3(80.0f, 300.0f, 100.0f), glm::normalize(glm::vec3(0.0f) - glm::vec3(80.0f, 300.0f, 100.0f)), 1,
            0.05f, 0.002f, 0.0f, glm::cos(glm::radians(30.0f)), glm::cos(glm::radians(35.0f))}
    },
    //noise
    {0.015f, 1.03f, 0.003f, 0.15f},
    //fog
    {glm::vec3(1.0f), 0.2f, 0.6f, 0.25f, 0.125f },
    //hdr
    2.2f, 0.25f,
    3000.0f,
    //camera
    glm::vec3(-1000.0f, 222.2f, 0.0f), 1.7f, 1.7f
},

    // B R E A K F A S T   R O O M 
Parameters{
    {
        //global directional light
        { glm::vec3(4.0f), glm::vec3(12.0f, 10.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -0.2f), 2 },
        //lamp 1
        { glm::vec3(2.0f, 2.0f, 1.3f), glm::vec3(1.0f, 4.15f, -1.92f), glm::vec3(0.001f, -1.0f, 0.0f), 1,
        0.025f, 0.01f, 0.0f, 1.055f, 0.72f },
        //lamp 2
        { glm::vec3(2.0f, 2.0f, 1.3f), glm::vec3(-2.15f, 4.15f, -1.92f), glm::vec3(0.001f, -1.0f, 0.0f), 1,
        0.025f, 0.01f, 0.0f, 1.055f, 0.72f },
        //lamp 3 as sun
        { glm::vec3(10.0f), glm::vec3(15.0f, 10.0f, -1.0f), glm::vec3(-0.6f, -0.33f, 0.0f), 3,
        0.025f, 0.01f, 0.0f, 1.0f, 0.92f },
    },
    //noise
    { 0.015f, 0.1f, 0.5f, 0.3f },
    //fog
    { glm::vec3(1.0f), 0.35f, 0.1f, 0.1f, 0.15f },
    //hdr
    2.2f, 0.15f,
    30.0f,
    //camera
    glm::vec3(2.6f, 4.8f, 7.0f), 3.8f, 1.25f
}, 

// S A N   M I G U E L
Parameters{
    {
        //global directional light
        { glm::vec3(20.0f), glm::vec3(0.0f, 25.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), 2 },
        //lamp 1
{ glm::vec3(2.0f, 2.0f, 1.3f), glm::vec3(1.0f, 4.15f, -1.92f), glm::vec3(0.001f, -1.0f, 0.0f), 1,
0.025f, 0.01f, 0.0f, 1.055f, 0.72f },
//lamp 2
{ glm::vec3(2.0f, 2.0f, 1.3f), glm::vec3(-2.15f, 4.15f, -1.92f), glm::vec3(0.001f, -1.0f, 0.0f), 1,
0.025f, 0.01f, 0.0f, 1.055f, 0.72f },
//lamp 3 as sun
{ glm::vec3(10.0f), glm::vec3(15.0f, 10.0f, -1.0f), glm::vec3(-0.6f, -0.33f, 0.0f), 3,
0.025f, 0.01f, 0.0f, 1.0f, 0.92f },
    },
    //noise
{ 0.015f, 0.1f, 0.5f, 0.3f },
//fog
{ glm::vec3(1.0f), 0.35f, 0.1f, 0.1f, 0.15f },
//hdr
2.2f, 0.15f,
30.0f,
//camera
glm::vec3(2.6f, 4.8f, 7.0f), 3.8f, 1.25f
}

};