#version 430

out vec4 fragmentColor;

// View and projection
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

// pixel in [-1,1]
in vec2 pixelPos;

// TODO use light ssbo
const vec3 LPos = vec3(5.0, 7.0, 5.0);

const float INFINITY = 1E+37;

// TODO make those GUI-updateable uniforms
const int maxSteps = 200; 	// max step count
const float eps = 1E-3;
const float shadowFactor = 32.0;

/*
** function definitions
*/
float opUnion(float d1, float d2);
float opIntersect(float d1, float d2);
float opComplement(float d);
float opSubtract(float d1, float d2);
// TODO add more operations
float opUnion(float d1, float d2)
{
	return min(d1, d2);
}

float opIntersect(float d1, float d2)
{
	return max(d1, d2);
}

float opComplement(float d)
{
	return -d;
}

float opSubtract(float d1, float d2)
{
	return max(d1, -d2);
}

// operations which return a "new p"
vec3 opRep( vec3 p, vec3 c)
{
    vec3 q = mod(p, c) - 0.5 * c;
    return q;
}

// TODO more objects
// distance functions containing translation
float dfSphere(vec3 p,vec3 c, float r);
float dfPBox(vec3 p, vec3 c, vec3 b);

// centered distance functions
float dfTorus(vec3 p, vec2 r);
float dfBox(vec3 p, vec3 b);
float dfPlane(vec3 p, vec3 q, vec3 n);


// distance functions containing translation
float dfSphere(vec3 p, vec3 c, float r)
{
	return length(p - c) - r;
}

float dfPBox(vec3 p, vec3 c, vec3 b)
{
	vec4 bv = vec4(b,0.0);
	vec2 n = vec2(1.0,0.0);

	float d = dfPlane(p, c+ bv.xww,n.xyy);
	d = opIntersect(d, dfPlane(p, c - bv.xww,-n.xyy));
	d = opIntersect(d, dfPlane(p, c + bv.wyw,n.yxy));
	d = opIntersect(d, dfPlane(p, c - bv.wyw,-n.yxy));

	d = opIntersect(d, dfPlane(p, c + bv.wwz,n.yyx));
	d = opIntersect(d, dfPlane(p, c - bv.wwz,-n.yyx));
	return d;
}

// centered distance functions
float dfTorus(vec3 p, vec2 r)
{
	vec2 t = vec2(length(p.xz) - r.x, p.y);
	return length(t) - r.y;
}

float dfBox(vec3 p, vec3 b)
{
	vec3 d = abs(p) - b;

	return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float dfPlane(vec3 p, vec3 q, vec3 n)
{
	
	return dot((p - q), n);
}

// TODO create an actual scene
float distancefield(vec3 p)
{
	//column
	float d1 = dfPBox(p, vec3(0.0, 0.0, 0.0), vec3(0.5, 1.5, 0.5));
	float d2 = dfSphere(p, vec3(0.0, 0.5, 0.0), 0.9);
	float d = opSubtract(d1, d2);

	// ground box
	float d3 = dfPBox(p, vec3(0.0, -2.8, 0.0), vec3(5.0, 0.2, 5.0));
	d = opUnion(d, d3);

	// addidtional sphere
	float sphere2 = dfSphere(p, vec3(3.0, -1.8, 0.0), 1.0);
	d = opUnion(d, sphere2);

	// torus
	float torus = dfTorus(p, vec2(2.0, 1.0));
	d = opUnion(d, torus);

	return d;
}

vec3 normal(vec3 p) {
	// compute discrete normal at point p
	vec2 epsilon = vec2(0.0005, 0.);

	vec3 n = vec3(
			distancefield(p + epsilon.xyy) - distancefield(p - epsilon.xyy),
			distancefield(p + epsilon.yxy) - distancefield(p - epsilon.yxy),
			distancefield(p + epsilon.yyx) - distancefield(p - epsilon.yyx));

	return normalize(n);
}

float raymarch(vec3 p, vec3 d) {
	float t = 0.0; 				// ray position
	int i = 0;					// current step count
	
	for (; i < maxSteps; i++) {
		vec3 point = p + t * d;
		float radius = distancefield(point); // Freier Bereich
		
		if (radius < eps) {
			break;				// Beende, falls nah genug
		}
		
		t += radius;			// Gehe weiter auf Strahl
	}
	
	// return position if an object was hit, infinity otherwise
	if (i == maxSteps) {
		return INFINITY;
	} else {
		return t;
	}
}

float ao(vec3 p, vec3 N, float delta, int steps)
{
	// calculate ambient occlusion
	float dif = 0.0;

	for(int i = 1; i <= steps; i++)
	{
		float maxDist = delta * float(steps);
		vec3 point = p + maxDist*N;

		float d = distancefield(point);

		dif += max(d/maxDist,0.0);
	}

	return dif/float(steps);
}

// generate a camera ray from pixel position and view/projection matrixes
void getCameraRay(in vec2 pxNDS, out vec3 from, out vec3 dir)
{
	mat4 PInv = inverse(ProjectionMatrix);
	mat4 VInv = inverse(ViewMatrix);

	vec3 pointNDS = vec3(pxNDS, -1.0);
	vec4 pointNDSH = vec4(pointNDS, 1.0);
	vec4 dirEye = PInv * pointNDSH;
	dirEye.w = 0.0;
	vec3 dirWorld = (VInv * dirEye).xyz;

	from = VInv[3].xyz;
	dir = normalize(dirWorld);
}

float softshadow( in vec3 shadePoint, in vec3 lightVec, float mint, float maxt)
{
    float res = 1.0;
    for( float t = mint; t < maxt;)
    {
		vec3 point = shadePoint + lightVec * t;
        float h = distancefield(point);

        if( h < 0.001 ) // full shadow
            return 0.0;

		// calculate penumbra factor
        res = min( res, shadowFactor * h / t );
        t += h;
    }
    return res;
}

// TODO add better shading model
// Simple shading
vec3 shade(vec3 p, vec3 V, vec3 N, vec3 color)
{
	vec3 l = normalize(LPos - p);
		
	vec3 lcol = vec3(0.0);
	float dotLN = dot(l, N);
	lcol += dotLN * color;

	vec3 R = reflect(-l, N);

	lcol += pow(max(dot(R, V), 0.0), 32.0) * float(dotLN > 0.0);

	// calculate shadow
	float shad = softshadow(p, l, 0.1, length(LPos - p) + 1);
	lcol *= shad;

	return lcol;
}

void main(){
   
	vec3 from = vec3(0.0);
	vec3 dir = vec3(0.0, 0.0, 1.0);

	getCameraRay(pixelPos, from, dir);

	float t = raymarch(from, dir);
  
	if(t < INFINITY)
	{
		vec3 p = from + t * dir;

		vec3 N = normal(p);

		vec3 c = shade(p, -dir, N, vec3(0.9, 0.9, 0.9));

		c *= ao(p, N, 0.05, 5);

		fragmentColor = vec4(c, 1.0);
	}
	else
	{
		fragmentColor = vec4(vec3(0.1), 1.0);
	}
	
	
}