#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include <memory>

#include "Utils/UtilCollection.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/VertexArray.h"
#include "Rendering/Uniform.h"

// no alignment needed
struct BlobSettings {
	glm::vec4 InnerColor;
	glm::vec4 OuterColor;
	float RadiusInner;
	float RadiusOuter;
};


int main(int argc, char* argv[]) {
	// init glfw, open window, manage context
	GLFWwindow* window = util::setupGLFWwindow(1600, 980, "Example 1");
	
	// init glew and check for errors
	util::initGLEW();

	// print OpenGL info
	util::printOpenGLInfo();

	util::enableDebugCallback();

	// get list of OpenGL extensions (can be searched later if needed)
	std::vector<std::string> extensions = util::getGLExtenstions();

	Shader vs("fuzzycircle.vert", GL_VERTEX_SHADER);
	Shader fs("fuzzycircle.frag", GL_FRAGMENT_SHADER);
	ShaderProgram sp(vs, fs);

	BlobSettings bs;
	bs.OuterColor = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
	bs.InnerColor = glm::vec4(1.0f, 1.0f, 0.75f, 1.0f);
	bs.RadiusInner = 0.25f;
	bs.RadiusOuter = 0.45f;

	// shader storage buffer object containg BlobSettings struct
	Buffer blobBuffer;
	blobBuffer.setData(std::vector<BlobSettings>{bs}, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
	blobBuffer.bindBase(0);

	std::array<float, 18> positionData = {
		// triangle 1
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		// triangle 2
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	std::array<float, 18> texCoordData = {
		// triangle 1
		0.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		// triangle 2
		0.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	Buffer vertexBuffer;
	vertexBuffer.setData(positionData, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

	Buffer texCoordBuffer;
	texCoordBuffer.setData(texCoordData, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

	VertexArray vao;
	vao.connectBuffer(vertexBuffer, 0, 3, GL_FLOAT, GL_FALSE);
	vao.connectBuffer(texCoordBuffer, 1, 3, GL_FLOAT, GL_FALSE);
	vao.bind();

	sp.use();

	// render loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		glClear(GL_COLOR_BUFFER_BIT);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
	}

	// close window
	glfwDestroyWindow(window);
}
