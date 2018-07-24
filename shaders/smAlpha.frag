#version 430
#extension GL_ARB_bindless_texture : require

flat in uint passDrawID;
in vec3 passTexCoord;

#include "common/material.glsl"

void main()
{             
	int materialIndex = int(materialIndices[passDrawID]);
    Material currentMaterial = materials[materialIndex];

	float alpha = 1.0f;

    if (currentMaterial.opacity == -1.0f) // has opacity texture instead of opacity
        alpha = texture(currentMaterial.opacityTexture, passTexCoord.rg).r;
    else if (currentMaterial.opacity == -2.0f)
        alpha = getDiffTextureAlpha(materialIndex);
    else
        alpha = currentMaterial.opacity;

	if(alpha <= 0.8f)
		discard;
}  