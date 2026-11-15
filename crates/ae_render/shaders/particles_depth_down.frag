#version 460
// The scene depth at half resolution for the particles' reduced-resolution
// target: per 2x2 block the farthest depth (infinite reverse-Z: the
// smallest value). The particles soften against it, so near a silhouette
// a block keeps the smoke behind the silhouette, and the composite gives
// it only to the full-resolution pixels at that depth.

layout(set = 1, binding = 0) uniform sampler2D depthTex;

layout(location = 0) out float outDepth;

void main() {
    ivec2 size = textureSize(depthTex, 0) - 1;
    ivec2 p = ivec2(gl_FragCoord.xy) * 2;
    float d = texelFetch(depthTex, min(p, size), 0).r;
    d = min(d, texelFetch(depthTex, min(p + ivec2(1, 0), size), 0).r);
    d = min(d, texelFetch(depthTex, min(p + ivec2(0, 1), size), 0).r);
    d = min(d, texelFetch(depthTex, min(p + ivec2(1, 1), size), 0).r);
    outDepth = d;
}
