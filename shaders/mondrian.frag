// Mondrian Image
uniform vec2 u_resolution;

vec3 smoothBorder(in vec2 st, in float start, in float end) 
{
    float left = smoothstep(start, end, st.x);
    float bottom = smoothstep(start, end,st.y);
    float right = smoothstep(start, end, 1.0-st.x);
    float top = smoothstep(start, end, 1.0-st.y);
    return vec3(left * bottom * right * top);
}

vec3 solidBorder(in vec2 st, in float borderWidth) 
{
    float left = step(borderWidth, st.x);
    float bottom = step(borderWidth, st.y);
    float right = step(borderWidth, 1.0-st.x);
    float top = step(borderWidth, 1.0-st.y);
    return vec3(left * bottom * right * top);
}

vec3 positionedRectangle(in vec2 st, in float borderWidth, in vec4 borderPositions) 
{
    float left = 1.0;
    float bottom = 1.0;
    float right = 1.0;
    float top = 1.0;
    if(st.x < borderPositions.y && st.x > borderPositions.x
		&& st.y < borderPositions.w && st.y > borderPositions.z) 
	{
        left = step(borderPositions.x + borderWidth, st.x);
        right = step(borderWidth, borderPositions.y - st.x);
        bottom = step(borderPositions.z + borderWidth, st.y);
        top = step(borderWidth, borderPositions.w-st.y);
    }
    return vec3(left * bottom * right * top);
}

vec3 positionedSmoothRectangle(in vec2 st, in float start, in float end, in vec4 borderPositions) 
{
    float left = 1.0;
    float bottom = 1.0;
    float right = 1.0;
    float top = 1.0;
    if(st.x < borderPositions.y && st.x > borderPositions.x
        && st.y < borderPositions.w && st.y > borderPositions.z)
	{
        left = smoothstep(borderPositions.x + start, borderPositions.x + end, st.x);
        right = smoothstep(start, end, borderPositions.y - st.x);
        bottom = smoothstep(borderPositions.z + start, borderPositions.z + end, st.y);
        top = smoothstep(start, end, borderPositions.w-st.y);
    }
    return vec3(left * bottom * right * top);
}

vec3 TDline(in vec2 st, in float width, in float xpos) 
{
    float line = 1.0;
    if(st.x < xpos && st.x > xpos - width) 
	{
        line = step(xpos + width, st.x);
    }
    return vec3(line);
}

vec3 RLline(in vec2 st, in float width, in float ypos) 
{
    float line = 1.0;
    if(st.y < ypos && st.y > ypos - width) 
	{
        line = step(width, ypos - st.y);
    }
    return vec3(line);
}

vec3 ClampedRLline(in vec2 st, in float width, in float ypos, in float minX, in float maxX) 
{
    float line = 1.0;
    if(st.y < ypos && st.y > ypos - width
		&& st.x > minX && st.x < maxX) 
	{
        line = step(width, ypos - st.y);
    }
    return vec3(line);
}

vec3 ClampedTDline(in vec2 st, in float width, in float xpos, in float minY, in float maxY) 
{
    float line = 1.0;
    if(st.x < xpos && st.x > xpos - width
		&& st.y > minY && st.y < maxY) 
	{
        line = step(xpos + width, st.x);
    }
    return vec3(line);
}

void main()
{
    vec2 st = gl_FragCoord.xy/u_resolution.xy;
    vec3 color = vec3(0.9, 0.9, 0.8);

    vec4 borderPos1 = vec4(0.2, 0.4, 0.2, 0.4);
    vec4 borderPos2 = vec4(0.3, 0.5, 0.3, 0.5);

    if(st.x < 0.2 && st.y < 1.0 && st.y > 0.7)
	{
        color = vec3(0.8, 0.2, 0.2);
    }

    if(st.x > 0.95 && st.y > 0.7)
	{
        color = vec3(1.0, 0.8, 0.2);
    }

    if(st.x > 0.8 && st.y < 0.1)
	{
        color = vec3(0.2, 0.2, 1.0);
    }

    //color = smoothRectangle(st, 0.01, 0.05);
    //color *= positionedRectangle(st, 0.02, borderPos1);
    //color *= positionedRectangle(st, 0.02, borderPos2);

    color *= TDline(st, 0.02, 0.2);
    color *= TDline(st, 0.02, 0.8);
    color *= TDline(st, 0.02, 0.95);

    color *= RLline(st, 0.02, 0.85);
    color *= RLline(st, 0.02, 0.7);
    color *= ClampedRLline(st, 0.02, 0.1, 0.2, 1.0);
    color *= ClampedTDline(st, 0.02, 0.07, 0.7, 1.0);


    gl_FragColor = vec4(color,1.0);
}