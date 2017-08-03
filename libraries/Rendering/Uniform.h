#pragma once

#include <string>

template<typename T>
class Uniform {
public:
	Uniform(const std::string& name, T content) :
		m_name(name), m_content(content) {};

    /**
	 * \brief returns the uniform name
	 * \return uniform name
	 */
	const std::string& getName() const;

    /**
     * \brief returns the current content
     * \return current content
     */
    T getContent() const;

    /**
     * \brief returns the 'content-has-been-changed'-flag
     * \return change flag
     */
    bool getChangeFlag() const;

    /**
	 * \brief sets the (cpu-sided) content and the change flag
	 * \param content 
	 */
	void setContent(const T &content);

    /**
     * \brief resets the change flag
     */
    void hasBeenUpdated();

private:
    bool m_hasChanged = true;
	std::string m_name;
	T m_content;
};

template<typename T>
const std::string& Uniform<T>::getName() const {
	return m_name;
}

template<typename T>
bool Uniform<T>::getChangeFlag() const {
    return m_hasChanged;
}

template<typename T>
void Uniform<T>::hasBeenUpdated() {
    m_hasChanged = false;
}

template<typename T>
T Uniform<T>::getContent() const {
    return m_content;
}

template<typename T>
void Uniform<T>::setContent(const T &content) {
    m_hasChanged = true;
	m_content = content;
}