#version 430

layout (location = 0) in vec4 position;

out vec2 pixelPos;

void main(){

   gl_Position =  position;
   pixelPos = position.xy;

}