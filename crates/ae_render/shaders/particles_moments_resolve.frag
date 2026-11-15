#version 460
// Moment-based OIT, resolve: the media's transmittance-weighted colour,
// normalised and scaled by their total opacity (1 - exp(-b0)), plus the
// emissive light; premultiplied, for the composite.
layout(set = 1, binding = 2) uniform sampler2D momentB0;
layout(set = 1, binding = 4) uniform sampler2D media;
layout(set = 1, binding = 5) uniform sampler2D emissive;

layout(location = 0) out vec4 outColor;

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    float opacity = 1.0 - exp(-texelFetch(momentB0, at, 0).r);
    vec4 m = texelFetch(media, at, 0);
    vec3 rgb = m.a > 1e-6 ? m.rgb / m.a * opacity : vec3(0.0);
    outColor = vec4(rgb + texelFetch(emissive, at, 0).rgb, opacity);
}
