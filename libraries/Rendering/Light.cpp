#include "Light.h"
#include <glm/gtc/matrix_transform.inl>
#include "imgui/imgui.h"
#include <glm/gtc/type_ptr.hpp>
#include <sstream>
#include "Cubemap.h"

using namespace gl;

Light::Light(glm::vec3 color, glm::vec3 direction, glm::ivec2 shadowMapRes) // DIRECTIONAL
: m_type(LightType::directional), m_shadowMapRes(shadowMapRes),
m_shadowTexture(std::make_shared<Texture>(GL_TEXTURE_2D, GL_LINEAR, GL_LINEAR)), m_shadowMapFBO(GL_DEPTH_ATTACHMENT, *m_shadowTexture),
m_genShadowMapProgram("lightTransform.vert", "nothing.frag", BufferBindings::g_definitions)
{
    // TODO USE POSITION TOO, FOR SHADOW MAPS
    checkParameters();

    // init shadowMap
    m_shadowTexture->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_shadowTexture->getName(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(m_shadowTexture->getName(), GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    m_shadowTexture->initWithoutData(m_shadowMapRes.x, m_shadowMapRes.y, GL_DEPTH_COMPONENT32F);

    m_modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", glm::mat4(1.0f));
    m_lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", glm::mat4(1.0f));

    m_genShadowMapProgram.addUniform(m_modelUniform);
    m_genShadowMapProgram.addUniform(m_lightSpaceUniform);

    // init gpu struct
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = color;
    m_gpuLight.direction = direction;
    m_gpuLight.shadowMap = m_shadowTexture->generateHandle();

    recalculateLightSpaceMatrix();
}

Light::Light(glm::vec3 color, glm::vec3 position, float constant, float linear, float quadratic, glm::ivec2 shadowMapRes) // POINT
: m_type(LightType::point), m_shadowMapRes(shadowMapRes),
m_shadowTexture(std::make_shared<Cubemap>(GL_LINEAR, GL_LINEAR)), m_shadowMapFBO(GL_DEPTH_ATTACHMENT, *m_shadowTexture),
m_genShadowMapProgram({
    Shader("transform.vert", GL_VERTEX_SHADER, BufferBindings::g_definitions),
    Shader("omnidirectional.geom", GL_GEOMETRY_SHADER, BufferBindings::g_definitions),
    Shader("omnidirectional.frag", GL_FRAGMENT_SHADER, BufferBindings::g_definitions) })
{
    checkParameters();

    // TODO THIS IS TEMPORARY; IMPLEMENT OMNIDIR. SHADOW MAPS
    // init shadowMap 
    auto currentCubemap = std::static_pointer_cast<Cubemap>(m_shadowTexture);
    currentCubemap->initWithoutData(m_shadowMapRes.x, m_shadowMapRes.y, GL_DEPTH_COMPONENT32F, GL_RED, GL_FLOAT, 6);

    m_lightPosUniform = std::make_shared<Uniform<glm::vec3>>("lightPos", position);

    m_modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", glm::mat4(1.0f));
    m_lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", glm::mat4(1.0f));

    m_genShadowMapProgram.addUniform(m_modelUniform);
    m_genShadowMapProgram.addUniform(m_lightSpaceUniform);
    m_genShadowMapProgram.addUniform(m_lightPosUniform);

    // init gpu struct
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = color;
    m_gpuLight.position = position;
    m_gpuLight.constant = constant;
    m_gpuLight.linear = linear;
    m_gpuLight.quadratic = quadratic;
    m_gpuLight.shadowMap = currentCubemap->generateHandle();
    recalculateLightSpaceMatrix();

    std::cout << "Shadow maps for point lights are not supported yet\n";
    //throw std::runtime_error("POINT LIGHTS NOT SUPPORTED YET");
}

Light::Light(glm::vec3 color, glm::vec3 position, glm::vec3 direction, float constant, float linear, float quadratic, float cutOff, float outerCutOff, glm::ivec2 shadowMapRes) // SPOT
    : m_type(LightType::spot), m_shadowMapRes(shadowMapRes),
    m_shadowTexture(std::make_shared<Texture>(GL_TEXTURE_2D, GL_LINEAR, GL_LINEAR)), m_shadowMapFBO(GL_DEPTH_ATTACHMENT, *m_shadowTexture),
   m_genShadowMapProgram("lightTransform.vert", "nothing.frag", BufferBindings::g_definitions)
{
    checkParameters();

    // init shadowMap
    m_shadowTexture->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_shadowTexture->getName(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(m_shadowTexture->getName(), GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    m_shadowTexture->initWithoutData(m_shadowMapRes.x, m_shadowMapRes.y, GL_DEPTH_COMPONENT32F);

    m_modelUniform = std::make_shared<Uniform<glm::mat4>>("ModelMatrix", glm::mat4(1.0f));
    m_lightSpaceUniform = std::make_shared<Uniform<glm::mat4>>("lightSpaceMatrix", glm::mat4(1.0f));

    m_genShadowMapProgram.addUniform(m_modelUniform);
    m_genShadowMapProgram.addUniform(m_lightSpaceUniform);

    // init gpu struct
    m_gpuLight.type = static_cast<int>(m_type);
    m_gpuLight.color = color;
    m_gpuLight.position = position;
    m_gpuLight.direction = direction;
    m_gpuLight.constant = constant;
    m_gpuLight.linear = linear;
    m_gpuLight.quadratic = quadratic;
    m_gpuLight.cutOff = cutOff;
    m_gpuLight.outerCutOff = outerCutOff;
    m_gpuLight.shadowMap = m_shadowTexture->generateHandle();

    recalculateLightSpaceMatrix();
}


void Light::renderShadowMap(const std::vector<std::shared_ptr<Mesh>>& scene)
{
    if (!m_hasShadowMap)
        return;

    recalculateLightSpaceMatrix();

    //store old viewport
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    //set sm settings
    m_genShadowMapProgram.use();
    glViewport(0, 0, m_shadowMapRes.x, m_shadowMapRes.y);
    m_shadowMapFBO.bind();
    glClear(GL_DEPTH_BUFFER_BIT);
    glCullFace(GL_FRONT);

    if (m_type == LightType::point && m_lightPosUniform->getContent() != m_gpuLight.position)
        m_lightPosUniform->setContent(m_gpuLight.position);

    //render scene
    std::for_each(scene.begin(), scene.end(), [&](auto& mesh)
    {
        m_modelUniform->setContent(mesh->getModelMatrix()); // TODO change shaders to use model matrix buffer instead
        m_genShadowMapProgram.updateUniforms();
        mesh->forceDraw(); // TODO CULL
    });

    //restore previous rendering settings
    m_shadowMapFBO.unbind();
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glCullFace(GL_BACK);
}

void Light::recalculateLightSpaceMatrix()
{
    if (m_type == LightType::directional)
    {
        const float nearPlane = 3.0f, farPlane = 3000.0f;
        m_lightProjection = glm::ortho(-2000.0f, 2000.0f, -2000.0f, 2000.0f, nearPlane, farPlane);

        glm::vec3 up(0.0f, 1.0f, 0.0f);

        if (glm::length(glm::cross(m_gpuLight.position + m_gpuLight.direction, up)) < 0.01f)
        {
            up = glm::vec3(1.0f, 0.0f, 0.0f);
        }

        m_lightView = glm::lookAt(m_gpuLight.position,
            m_gpuLight.position + m_gpuLight.direction,
            up);
    }
    else if (m_type == LightType::spot) 
    {
        const float nearPlane = 3.0f, farPlane = 1000.0f;
        // NOTE: ACOS BECAUSE CUTOFF HAS COS BAKED IN
        m_lightProjection = glm::perspective(2.0f*glm::acos(m_gpuLight.outerCutOff), static_cast<float>(m_shadowMapRes.x) / static_cast<float>(m_shadowMapRes.y), nearPlane, farPlane);

        m_lightView = glm::lookAt(m_gpuLight.position,
            m_gpuLight.position + m_gpuLight.direction, // aimed at the center
            glm::vec3(0.0f, 1.0f, 0.0f));
    }
    else if (m_type == LightType::point)
    {
        const float nearPlane = 3.0f, farPlane = 1000.0f;
        // TODO is cutoff supposed to be used here? or 90 degrees (cube)?
        m_lightProjection = glm::perspective(glm::radians(90.0f), static_cast<float>(m_shadowMapRes.x) / static_cast<float>(m_shadowMapRes.y), nearPlane, farPlane);
        m_lightView = glm::mat4(1.0f); // calculate finished matrix in shader
    }

    m_gpuLight.lightSpaceMatrix = m_lightProjection * m_lightView;
    m_lightSpaceUniform->setContent(m_gpuLight.lightSpaceMatrix);
}

void Light::checkParameters()
{
    // TODO implement this if there are constructors that do not initalize all the params for the specific type
}

const GPULight& Light::getGpuLight() const
{
    return m_gpuLight;
}

void Light::setPosition(glm::vec3 pos)
{
    m_gpuLight.position = pos;
    recalculateLightSpaceMatrix();
}

void Light::setColor(glm::vec3 col)
{
    m_gpuLight.color = col;
}

void Light::setCutoff(float cutoff)
{
    m_gpuLight.cutOff = cutoff;
    recalculateLightSpaceMatrix();
}

void Light::setDirection(glm::vec3 dir)
{
    m_gpuLight.direction = dir;
    recalculateLightSpaceMatrix();
}


void Light::setConstant(float constant)
{
    m_gpuLight.constant = constant;
}

float Light::getConstant() const
{
    return m_gpuLight.constant;
}

void Light::setLinear(float linear)
{
    m_gpuLight.linear = linear;
}

float Light::getLinear() const
{
    return m_gpuLight.linear;
}

void Light::setQuadratic(float quadratic)
{
    m_gpuLight.quadratic = quadratic;
}

float Light::getQuadratic() const
{
    return m_gpuLight.quadratic;
}

glm::vec3 Light::getPosition() const
{
    return m_gpuLight.position;
}

glm::vec3 Light::getColor() const
{
    return m_gpuLight.color;
}

float Light::getCutoff() const
{
    return m_gpuLight.cutOff;
}

void Light::setOuterCutoff(float cutOff)
{
    m_gpuLight.outerCutOff = cutOff;
}

float Light::getOuterCutoff() const
{
    return m_gpuLight.outerCutOff;
}

LightType Light::getType() const
{
    return m_type;
}

glm::vec3 Light::getDirection() const
{
    return m_gpuLight.direction;
}


bool Light::showLightGUI(const std::string& name)
{
    ImGui::Begin("Light GUI");
    bool lightChanged = showLightGUIContent(name);
    ImGui::End();

    if (lightChanged) recalculateLightSpaceMatrix();
    // TODO only recalculate if necessary

    return lightChanged;
}

bool Light::showLightGUIContent(const std::string& name)
{
    std::array<std::string, 3> lightTypeNames = { "Directional", "Point", "Spot" };

    ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
    std::stringstream fullName;
    fullName << name << " (Type: " << lightTypeNames[static_cast<int>(m_type)] << ")";
    bool lightChanged = false;
    ImGui::Text(fullName.str().c_str());
    if (ImGui::SliderFloat3((std::string("Color ") + name).c_str(), value_ptr(m_gpuLight.color), 0.0f, 1.0f))
    {
        lightChanged = true;
    }
    if (m_type == LightType::directional || m_type == LightType::spot)
    {
        if (ImGui::SliderFloat3((std::string("Direction ") + name).c_str(), value_ptr(m_gpuLight.direction), -1.0f, 1.0f))
        {
            lightChanged = true;
        }
    }
    if (m_type == LightType::spot)
    {
        if (ImGui::SliderFloat((std::string("Cutoff ") + name).c_str(), &m_gpuLight.cutOff, 0.0f, glm::radians(90.0f)))
        {
            lightChanged = true;
			if (m_gpuLight.cutOff < m_gpuLight.outerCutOff)
				m_gpuLight.outerCutOff = m_gpuLight.cutOff - 0.001;
        }
        if (ImGui::SliderFloat((std::string("Outer cutoff ") + name).c_str(), &m_gpuLight.outerCutOff, 0.0f, glm::radians(90.0f)))
        {
            lightChanged = true;
			if (m_gpuLight.cutOff < m_gpuLight.outerCutOff)
				m_gpuLight.cutOff = m_gpuLight.outerCutOff + 0.001;
        }
    }
    if (m_type == LightType::spot || m_type == LightType::point)
    {
        if (ImGui::SliderFloat3((std::string("Position ") + name).c_str(), value_ptr(m_gpuLight.position), -500.0f, 500.0f))
        {
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Constant ") + name).c_str(), &m_gpuLight.constant, 0.0f, 1.0f))
        {
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Linear ") + name).c_str(), &m_gpuLight.linear, 0.0f, 0.25f))
        {
            lightChanged = true;
        }
        if (ImGui::SliderFloat((std::string("Quadratic ") + name).c_str(), &m_gpuLight.quadratic, 0.0f, 0.1f))
        {
            lightChanged = true;
        }
    }
    return lightChanged;
}