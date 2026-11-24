// What the splat vertex shader hands the raw and churn stamps.
layout(location = 0) in vec2 corner;      // -1..1 across the quad
layout(location = 1) flat in vec4 state;  // wave amplitude, churn, whitewater, age (s)
layout(location = 2) flat in vec2 stamp;  // x: 1 for a blast's splash stamp; y: its age in T
layout(location = 0) out vec4 color;
