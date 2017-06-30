#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>

template<typename T>
class Uniform {
public:
	Uniform(std::string name, T content) :
		m_name(name), m_content(content) {};

	std::string getName();
	T getContent();
	void setContent(T &content);

private:
	std::string m_name;
	T m_content;
};

template<typename T>
std::string Uniform<T>::getName() {
	return m_name;
}

template<typename T>
T Uniform<T>::getContent() {
	return m_content;
}

template<typename T>
void Uniform<T>::setContent(T &content) {
	m_content = content;
}