# OpenGL-Playground

OpenGL project to test things and to implement things from the OpenGL 4 Cookbook (improved) / CG lectures / own ideas

OpenGL-Version:
* 4.5 and up should work

Used libraries:
* ImGui: https://github.com/ocornut/imgui
* stb_image_write: https://github.com/nothings/stb
* Assimp (tested with 3.3.1)
* glfw3 (tested with 3.2.1)
* glew (tested with 2.0.0)
* glm (tested with 0.9.8.4)

### demo 1: lighting
* work in progress
* phong lighting (spotlights, directional lights, specular & diffuse, fog)
* toon shading
* GUI-controllable lighting parameters
* Shader live reloading

### demo 2: distance fields
* work in progress
* raymarching distance fields: soft shadows, simple distance field ambient occlusion
* Shader live reloading
* heavily inspired by http://www.iquilezles.org/, university lectures

### demo 3: shaders
* work in progress
* testing some shader-only drawing stuff, "the book of shaders"-style

### demo 4: color palette
* work in progress
* goal: extract a color palette from an image using k-means on GPU
* requires ARB_bindless texture
* uses image load/store
