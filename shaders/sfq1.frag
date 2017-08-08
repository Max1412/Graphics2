uniform vec2 u_resolution;
//uniform vec2 u_mouse;
uniform float u_time;

// Plot a line on Y using a value between 0.0-1.0
float plot(vec2 st, float pct){
    return smoothstep( pct-0.02, pct, st.y) -
        smoothstep( pct, pct+0.02, st.y);
}

void main() {
    vec2 st = gl_FragCoord.xy/u_resolution;
    float b = 0.0;
    if(st.x < 0.5){
        b = 1.0;
    } else {
        b = 0.5;
    }
    gl_FragColor = vec4(abs(sin(u_time)), b, 0.0, 1.0);
}