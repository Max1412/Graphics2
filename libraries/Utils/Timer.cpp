#include "Timer.h"
#include "imgui/imgui.h"

Timer::Timer()
{
    glCreateQueries(GL_TIME_ELAPSED, 1, &m_query);
}

Timer::~Timer()
{
    if (glfwGetCurrentContext() != nullptr)
    {
        glDeleteQueries(1, &m_query);
    }
    util::getGLerror(__LINE__, __FUNCTION__);
}

void Timer::start() const
{
    glBeginQuery(GL_TIME_ELAPSED, m_query);
}

void Timer::stop()
{
    glEndQuery(GL_TIME_ELAPSED);
    while (!m_done)
    {
        glGetQueryObjectiv(m_query, GL_QUERY_RESULT_AVAILABLE, &m_done);
    }
    glGetQueryObjectuiv(m_query, GL_QUERY_RESULT, &m_elapsedTime);
    m_ftimes.push_back(m_elapsedTime / 1000000.f);
    if (m_ftimes.size() > 1000)
    {
        m_ftimes.erase(m_ftimes.begin());
    }
    util::getGLerror(__LINE__, __FUNCTION__);
}

void Timer::drawGuiWindow(GLFWwindow* window)
{
    ImGui::SetNextWindowSize(ImVec2(300, 100), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Performance");
	drawGuiContent(window);
    ImGui::End();
}

void Timer::drawGuiContent(GLFWwindow* window, bool compact)
{
	auto flaccTime = 0.0f;
	if (m_ftimes.size() > 21)
	{
		for (auto i = m_ftimes.size() - 21; i < m_ftimes.size(); ++i)
		{
			flaccTime += m_ftimes.at(i);
		}
		flaccTime /= 20.0f;
	}
	//show everything in one line
	if (compact)
	{
		const float availableWidth = ImGui::GetContentRegionAvailWidth();
		if (availableWidth >= 300)
			ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - 300); 
		else if (availableWidth >= 70)
			ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - 70);

		ImGui::PushItemWidth(70);
		ImGui::Text("%.3f ms", flaccTime);
		if (ImGui::IsItemClicked()) util::saveFBOtoFile("demo1", window);
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Click to save FBO");
			ImGui::EndTooltip();
		}
		//don't show plot if not enough space
		if (availableWidth >= 300)
		{
			ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - 230);
			ImGui::PushItemWidth(240);
			const auto offset = static_cast<int>(m_ftimes.size() <= 240 ? m_ftimes.size() - 1 : 240);
			ImGui::PlotLines("", &m_ftimes.back() - offset, offset, 0, nullptr, 0.0f, std::numeric_limits<float>::max());
		}
	}
	else
	{
		ImGui::PlotLines("Frametime", m_ftimes.data(), static_cast<int>(m_ftimes.size()), 0, nullptr, 0.0f, std::numeric_limits<float>::max());
		if (ImGui::Button("Save FBO"))
		{
			util::saveFBOtoFile("demo1", window);
		}
		ImGui::SameLine();
		ImGui::Value("Frametime (milliseconds)", flaccTime);
	}
}