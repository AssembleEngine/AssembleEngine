#version 460
// Moment-based OIT, pass 2: each fragment's colour, weighted by the
// transmittance in front of it reconstructed from the moments. Media and
// emissive light are summed apart (the resolve normalises the media).
#include "particles_common.glsl"
#include "particles_frag.glsl"

layout(set = 1, binding = 2) uniform sampler2D momentB0;
layout(set = 1, binding = 3) uniform sampler2D moments;

layout(location = 0) out vec4 outMedia;
layout(location = 1) out vec4 outEmissive;

// Peters & Klein, "Moment-based order-independent transparency" (2018),
// supplementary code: the transmittance at `depth` from four power
// moments normalised by b0, biased towards a valid moment vector.
float transmittance(float b0, vec4 b, float depth) {
    const float bias = 5e-7;
    const float overestimation = 0.25;
    b = mix(b, vec4(0.0, 0.375, 0.0, 0.375), bias);
    float L21D11 = -b.x * b.y + b.z;
    float D11 = -b.x * b.x + b.y;
    float InvD11 = 1.0 / D11;
    float L21 = L21D11 * InvD11;
    float SquaredDepthVariance = -b.y * b.y + b.w;
    float D22 = -L21D11 * L21 + SquaredDepthVariance;
    vec3 c = vec3(1.0, depth, depth * depth);
    c.y -= b.x;
    c.z -= b.y + L21 * c.y;
    c.y *= InvD11;
    c.z /= D22;
    c.y -= L21 * c.z;
    c.x -= dot(c.yz, b.xy);
    float p = c.y / c.z;
    float q = c.x / c.z;
    float r = sqrt(max(p * p * 0.25 - q, 0.0));
    vec3 z = vec3(depth, -p * 0.5 - r, -p * 0.5 + r);
    vec3 f = vec3(overestimation, z.y < z.x ? 1.0 : 0.0, z.z < z.x ? 1.0 : 0.0);
    float f01 = (f.y - f.x) / (z.y - z.x);
    float f12 = (f.z - f.y) / (z.z - z.y);
    float f012 = (f12 - f01) / (z.z - z.x);
    vec3 poly;
    poly.x = f012;
    poly.y = poly.x;
    poly.x = f01 - poly.x * z.y;
    poly.z = poly.y;
    poly.y = poly.x - poly.y * z.x;
    poly.x = f.x - poly.x * z.x;
    float absorbance = poly.x + dot(b.xy, poly.yz);
    return clamp(exp(-b0 * absorbance), 0.0, 1.0);
}

void main() {
    float front;
    float k = coverage(front);
    if (k <= 0.0) {
        discard;
    }
    ivec2 at = ivec2(gl_FragCoord.xy);
    float b0 = texelFetch(momentB0, at, 0).r;
    float t = 1.0;
    if (b0 > 1e-5) {
        t = transmittance(b0, texelFetch(moments, at, 0) / b0, warpDepth(front));
    }
    vec4 c = inColor * k * t;
    if (inColor.a > 0.0) {
        outMedia = c;
        outEmissive = vec4(0.0);
    } else {
        outMedia = vec4(0.0);
        outEmissive = vec4(c.rgb, 0.0);
    }
}
