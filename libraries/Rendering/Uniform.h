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
	void setContent(const T &content);
    bool getChangeFlag();
    void hasBeenUpdated();

private:
    bool m_hasChanged = true;
	std::string m_name;
	T m_content;
};

template<typename T>
std::string Uniform<T>::getName() {
	return m_name;
}

template<typename T>
bool Uniform<T>::getChangeFlag() {
    return m_hasChanged;
}

template<typename T>
void Uniform<T>::hasBeenUpdated() {
    m_hasChanged = false;
}

template<typename T>
T Uniform<T>::getContent() {
	return m_content;
}

template<typename T>
void Uniform<T>::setContent(const T &content) {
    m_hasChanged = true;
	m_content = content;
}