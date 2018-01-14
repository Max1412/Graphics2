#version 430
#extension GL_ARB_bindless_texture : require

out vec4 FragColor;

in vec3 localPos;
  
layout(binding = 6, std430) buffer skyboxBuffer
{
    samplerCube environmentMap;
};
  
void main()
{
    vec3 envColor = texture(environmentMap, localPos).rgb;
    //vec3 envColor = textureLod(environmentMap, localPos, 1.2).rgb; 

    
    envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0/2.2)); 
  
    FragColor = vec4(envColor, 1.0);
}