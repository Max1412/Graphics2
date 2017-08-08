#include "Timer.h"
#include "imgui/imgui.h"

Timer::Timer() {
    glGenQueries(1, &m_query);
}

void Timer::del() {
    glDeleteQueries(1, &m_query);
}

void Timer::start() {
    glBeginQuery(GL_TIME_ELAPSED, m_query);
}

void Timer::stop() {
    glEndQuery(GL_TIME_ELAPSED);
    while (!m_done) {
        glGetQueryObjectiv(m_query, GL_QUERY_RESULT_AVAILABLE, &m_done);
    }
    glGetQueryObjectuiv(m_query, GL_QUERY_RESULT, &m_elapsedTime);
    m_ftimes.push_back(m_elapsedTime / 1000000.f);
    if (m_ftimes.size() > 1000) {
        m_ftimes.erase(m_ftimes.begin());
    }
    util::getGLerror(__LINE__, __FUNCTION__);
}

void Timer::drawGuiWindow(GLFWwindow* window) {
    ImGui::SetNextWindowSize(ImVec2(300, 100), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Performance");
    ImGui::PlotLines("Frametime", m_ftimes.data(), m_ftimes.size(), 0, nullptr, 0.0f, std::numeric_limits<float>::max());
    float flaccTime = 0.0f;
    if (m_ftimes.size() > 21) {
        for (size_t i = m_ftimes.size() - 21; i < m_ftimes.size(); ++i) {
            flaccTime += m_ftimes.at(i);
        }
        flaccTime /= 20.0f;
    }
    ImGui::Value("Frametime (milliseconds)", flaccTime);
    if (ImGui::Button("Save FBO")) {
        util::saveFBOtoFile("demo1", window);
    }
    ImGui::End();
}