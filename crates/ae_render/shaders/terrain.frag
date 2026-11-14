#version 460
// Per-pixel normals come straight from the heightmap, so lighting does not depend on the
// mesh resolution: coarse triangles in flat areas are indistinguishable from fine ones.
#define GRAPHICS
#include "terrain_common.glsl"
#ifdef BARY
#extension GL_EXT_fragment_shader_barycentric : require
#endif

layout(location = 0) in vec3 inWorld;
layout(location = 1) flat in uint inDepth;
layout(location = 0) out vec4 outColor;

vec3 hue(float h) {
    return clamp(abs(fract(h + vec3(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

void main() {
    float cs = u.map.w;
    vec2 cell = inWorld.xz / cs;
    float hl = sampleHeight(cell - vec2(1, 0)), hr = sampleHeight(cell + vec2(1, 0));
    float hd = sampleHeight(cell - vec2(0, 1)), hu = sampleHeight(cell + vec2(0, 1));
    vec3 n = normalize(vec3(hl - hr, 2.0 * cs, hd - hu));

    float t = (inWorld.y - u.height.y) / max(u.height.x, 1e-3);
    float slope = 1.0 - n.y;

    vec3 grass = vec3(0.23, 0.33, 0.12);
    vec3 dry = vec3(0.42, 0.38, 0.22);
    vec3 rock = vec3(0.36, 0.33, 0.30);
    vec3 darkRock = vec3(0.22, 0.20, 0.19);
    vec3 snow = vec3(0.92, 0.94, 0.97);

    vec3 albedo = mix(grass, dry, smoothstep(0.08, 0.2, t));
    albedo = mix(albedo, mix(rock, darkRock, smoothstep(0.3, 0.8, t)), smoothstep(0.12, 0.3, slope));
    float snowLine = 0.55 + 0.1 * sin(inWorld.x * 0.0013) * cos(inWorld.z * 0.0011);
    albedo = mix(albedo, snow, smoothstep(snowLine, snowLine + 0.05, t) * (1.0 - smoothstep(0.35, 0.6, slope)));

    if ((u.cfg.z & FLAG_MATERIALS) != 0u) {
        // Per-cell gameplay materials (AssemblyEngine addition), blended
        // bilinearly between cell centres; slopes still turn to rock.
        // Placeholder until material tilesets.
        const vec3 palette[8] = vec3[](
            vec3(0.23, 0.33, 0.12), // grass
            vec3(0.10, 0.20, 0.07), // forest
            vec3(0.60, 0.54, 0.38), // sand
            vec3(0.36, 0.33, 0.30), // rock
            vec3(0.32, 0.30, 0.22), // water (bed)
            vec3(0.45, 0.41, 0.35), // road
            vec3(0.30, 0.23, 0.15), // mud
            vec3(1.0, 0.0, 1.0));
        vec2 p = cell - 0.5;
        ivec2 c0 = ivec2(floor(p));
        vec2 f = p - vec2(c0);
        ivec2 hi = ivec2(u.map.xy) - 1;
        vec3 m00 = palette[min(texelFetch(materialTex, clamp(c0, ivec2(0), hi), 0).r, 7u)];
        vec3 m10 = palette[min(texelFetch(materialTex, clamp(c0 + ivec2(1, 0), ivec2(0), hi), 0).r, 7u)];
        vec3 m01 = palette[min(texelFetch(materialTex, clamp(c0 + ivec2(0, 1), ivec2(0), hi), 0).r, 7u)];
        vec3 m11 = palette[min(texelFetch(materialTex, clamp(c0 + ivec2(1, 1), ivec2(0), hi), 0).r, 7u)];
        albedo = mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
        albedo = mix(albedo, rock, smoothstep(0.2, 0.45, slope));
    }

    if ((u.cfg.z & FLAG_LOD_COLORS) != 0u) {
        albedo = mix(albedo, hue(float(inDepth) * 0.13), 0.65);
    }

    vec3 L = normalize(u.sun.xyz);
    float diff = max(dot(n, L), 0.0);
    vec3 sky = vec3(0.45, 0.55, 0.7);
    vec3 color = albedo * (vec3(1.0, 0.95, 0.85) * 1.6 * diff + sky * (0.25 + 0.25 * n.y));

    // Ground under water (AssemblyEngine addition): tint by the depth of the
    // water column until a water pass exists (M7).
    if (u.water.y > 0.5 && inWorld.y < u.water.x) {
        float depth = u.water.x - inWorld.y;
        vec3 deep = vec3(0.04, 0.16, 0.24);
        vec3 shallow = vec3(0.16, 0.38, 0.42);
        vec3 tint = mix(shallow, deep, 1.0 - exp(-depth * u.water.z * 0.3));
        color = mix(color * vec3(0.6, 0.8, 0.85), tint, 1.0 - exp(-depth * u.water.z));
    }

    float dist = distance(inWorld, u.camPos.xyz);
    float fog = 1.0 - exp(-pow(dist * u.fog.w, 1.5));
    color = mix(color, u.fog.rgb, fog);

#ifdef BARY
    // Wireframe overlay. Sub-mesh triangles are emitted as (v0, apex, v2), so the
    // hypotenuse is the edge where the apex weight is 0: hiding it shows LEB quads.
    if ((u.cfg.z & (FLAG_WIRE_TRIS | FLAG_WIRE_QUADS)) != 0u) {
        vec3 b = gl_BaryCoordEXT;
        vec3 edge = smoothstep(vec3(0.0), fwidth(b) * 1.2, b);
        float e = (u.cfg.z & FLAG_WIRE_QUADS) != 0u ? min(edge.x, edge.z) : min(min(edge.x, edge.y), edge.z);
        color = mix(vec3(0.02), color, 0.35 + 0.65 * e);
    }
#endif
    outColor = vec4(color, 1.0);
}
