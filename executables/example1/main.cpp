#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>

#include "Utils/UtilCollection.h"
#include "Rendering/Shader.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Buffer.h"
#include "Rendering/VertexArray.h"

std::string GLubyteToString(const GLubyte* content) {
	return std::string(reinterpret_cast<const char*>(content));

}

int main(int argc, char* argv[]) {
	// init glfw, open window, manage context
	GLFWwindow* window = util::setupGLFWwindow(1600, 980, "Example 1");
	
	// init glew and check for errors
	util::initGLEW();

	// print OpenGL info
	util::printOpenGLInfo();

	// get list of OpenGL extensions (can be searched later if needed)
	std::vector<std::string> extensions = util::getGLExtenstions();

	Shader vs("basic.vert", GL_VERTEX_SHADER);
	Shader fs("basic.frag", GL_FRAGMENT_SHADER);
	ShaderProgram sp(vs, fs);
	sp.use();



	std::array<float, 9> positionData = {
		-0.8f, -0.8f, 0.0f,
		0.8f, -0.8f, 0.0f,
		0.0f, 0.8f, 0.0f };

	std::array<float, 9>  colorData = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f };

	Buffer positionBuffer;
	positionBuffer.setData(positionData, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

	Buffer colorBuffer;
	colorBuffer.setData(colorData, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

	VertexArray vao;
	vao.connectBuffer(positionBuffer, 0, 3, GL_FLOAT, GL_FALSE);
	vao.connectBuffer(colorBuffer, 1, 3, GL_FLOAT, GL_FALSE);
	vao.bind();

	// render loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		glDrawArrays(GL_TRIANGLES, 0, 3);
		glfwSwapBuffers(window);
	}

	// close window
	glfwDestroyWindow(window);
}
