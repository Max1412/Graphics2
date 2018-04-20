// time colors

uniform vec2 u_resolution;
uniform float u_time;

// Plot a line on Y using a value between 0.0-1.0
float plot(vec2 st, float pct)
{
    return smoothstep( pct-0.02, pct, st.y) -
        smoothstep( pct, pct+0.02, st.y);
}

void main() 
{
    vec2 st = gl_FragCoord.xy/u_resolution;
    gl_FragColor = vec4(abs(sin(u_time)), st.x, st.y, 1.0);
}