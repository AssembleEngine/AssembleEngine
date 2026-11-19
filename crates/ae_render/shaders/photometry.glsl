// Photometric lighting (src/lighting.rs): the scene's sun and sky tints,
// and the constant the Lambertian BRDF needs. The tints are normalised to
// unit luminance here (Rec. 709), so the illuminances the frame uniforms
// carry are the sun's and the sky's, not the tinted sums. Mirrors
// `lighting::SUN_TINT`, `lighting::SKY_TINT`.
#ifndef PHOTOMETRY_GLSL
#define PHOTOMETRY_GLSL

#define AE_PI 3.141592653589793
const vec3 AE_SUN_TINT = vec3(1.0, 0.95, 0.85);
const vec3 AE_SKY_TINT = vec3(0.45, 0.55, 0.70);

vec3 aeUnitTint(vec3 c) {
    return c / max(dot(c, vec3(0.2126, 0.7152, 0.0722)), 1e-6);
}

#endif
