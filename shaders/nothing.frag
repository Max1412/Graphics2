#version 430
layout(early_fragment_tests) in;

// "empty" fragment shader, used to only fill the depth buffer

void main()
{             
    // gl_FragDepth = gl_FragCoord.z;
}  