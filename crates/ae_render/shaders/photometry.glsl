// Photometric lighting (src/lighting.rs): the scene's sun and sky tints,
// the working space and the constant the Lambertian BRDF needs. The tints
// are authored in BT.709 and normalised to unit luminance there; when the
// scene's working space is BT.2020 (`u.photometry.y`, `WorkingSpace`) they
// are converted before the normalisation, so the illuminance scalar still
// is the illuminance. Mirrors `lighting::SUN_TINT`, `lighting::SKY_TINT`
// and `colorspace`.
#ifndef PHOTOMETRY_GLSL
#define PHOTOMETRY_GLSL

#define AE_PI 3.141592653589793
const vec3 AE_SUN_TINT = vec3(1.0, 0.95, 0.85);
const vec3 AE_SKY_TINT = vec3(0.45, 0.55, 0.70);
const vec3 AE_LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
const vec3 AE_LUMA_2020 = vec3(0.2627, 0.6780, 0.0593);
// BT.709 to BT.2020 (columns; src/colorspace.rs's matrix).
const mat3 AE_709_TO_2020 = mat3(
    0.6274039, 0.0690973, 0.0163914,
    0.3292830, 0.9195404, 0.0880133,
    0.0433131, 0.0113623, 0.8955953);

// A colour authored in BT.709 in the scene's working space.
vec3 aeToWorking(vec3 c, bool wide) {
    return wide ? AE_709_TO_2020 * c : c;
}

// `c` (BT.709) normalised to unit luminance in the working space.
vec3 aeUnitTint(vec3 c, bool wide) {
    vec3 w = aeToWorking(c, wide);
    return w / max(dot(w, wide ? AE_LUMA_2020 : AE_LUMA_709), 1e-6);
}

#endif
