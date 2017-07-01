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

	float angle = 0;

	glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f));
	
	auto rot = std::make_shared<Uniform<glm::mat4>>("RotationMatrix", rotationMatrix);
	
	sp.addUniform(rot);
	sp.use();

	// render loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		glClear(GL_COLOR_BUFFER_BIT);

		rot->setContent(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f)));
		sp.updateUniforms();


		glDrawArrays(GL_TRIANGLES, 0, 3);

		angle += 0.1f;

		glfwSwapBuffers(window);
	}

	colorBuffer.del();
	positionBuffer.del();
	vao.del();
	sp.del();

	// close window
	glfwDestroyWindow(window);
}
