#version 430

in vec3 passPosition;
in vec3 interpNormal;
flat in vec3 flatNormal;

struct Light
{
	vec4 pos; //pos.w=0 dir., pos.w=1 point light
	vec3 col;
	float spot_cutoff; //no spotlight if cutoff=0
	vec3 spot_direction;
	float spot_exponent;
};

struct Material
{
	vec3 diffColor;
	float kd;
	vec3 specColor;
	float ks;
	float shininess;
	float kt;
};

layout (std430, binding = 0) restrict readonly buffer LightBuffer {
	Light light[];
};

layout (std430, binding = 1) restrict readonly buffer MaterialBuffer {
	Material material[];
};

uniform mat4 ViewMatrix;
uniform vec3 lightAmbient;
uniform int matIndex;

uniform int useFlat;
uniform int useToon;
uniform int levels;

layout( location = 0 ) out vec4 fragmentColor;

void main() {
	vec3 passNormal = vec3(0.0, 0.0, 0.0);
	if(useFlat == 1){
		passNormal = flatNormal;
	} else {
		passNormal = interpNormal;
	}
	passNormal = normalize( passNormal );
	vec3 lightVector;
	float spot;
	vec3 diffuse_color;
	float diffuse_alpha;
	Material mat = material[ matIndex];
	diffuse_color = mat.diffColor;
	diffuse_alpha = 1.f - mat.kt;

    fragmentColor.rgb = mat.kd*diffuse_color*lightAmbient;

    if (useToon == 0) {

	    for ( int i = 0; i < light.length(); i++) {
		    vec3 light_camcoord = (ViewMatrix * light[i].pos).xyz;
		    if (light[i].pos.w > 0.001f)
			    lightVector = normalize( light_camcoord - passPosition);
		    else
			    lightVector = normalize(light_camcoord);
		    float cos_phi = max( dot( passNormal, lightVector), 0.000001f);

		    vec3 eye = normalize( -passPosition);
		    vec3 reflection = normalize( reflect( -lightVector, passNormal));
		    float cos_psi_n = pow( max( dot( reflection, eye), 0.000001f), mat.shininess);

		    if (light[i].spot_cutoff < 0.001f)
			    spot = 1.0;
		    else {
			    float cos_phi_spot = max( dot( -lightVector, normalize(mat3(ViewMatrix) * light[i].spot_direction)), 0.000001f);
			    if( cos_phi_spot >= cos( light[i].spot_cutoff))
				    spot = pow( cos_phi_spot, light[i].spot_exponent);
			    else
				    spot = 0.0f;
		    }
		    fragmentColor.rgb += mat.kd * spot * diffuse_color * cos_phi * light[i].col;
		    fragmentColor.rgb += mat.ks * spot * mat.specColor * cos_psi_n * light[i].col;
        }

    } else {

        for ( int i = 0; i < light.length(); i++) {
		    vec3 light_camcoord = (ViewMatrix * light[i].pos).xyz;
		    if (light[i].pos.w > 0.001f)
			    lightVector = normalize( light_camcoord - passPosition);
		    else
			    lightVector = normalize(light_camcoord);
		    float cos_phi = max( dot( passNormal, lightVector), 0.000001f);

            float scaleFactor = 1.0 / levels;
            vec3 diffuse = (diffuse_color * mat.kd) * floor( cos_phi * levels ) * scaleFactor;

            fragmentColor.rgb += light[i].col * diffuse;
         }
    }
	fragmentColor.a = diffuse_alpha;
}