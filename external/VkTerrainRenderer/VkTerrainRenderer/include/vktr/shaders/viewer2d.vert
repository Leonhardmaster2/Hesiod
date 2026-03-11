#version 460

// Fullscreen triangle — no vertex buffer needed
// Uses gl_VertexIndex 0,1,2 to cover the screen

layout(push_constant) uniform PushConstants {
    float aspect_ratio;
    float zoom;
    vec2  offset;
};

layout(location = 0) out vec2 frag_uv;

void main()
{
    // Generate fullscreen triangle vertices
    vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);

    // UV: map from clip space [-1,1] to [0,1], apply zoom and offset
    frag_uv = (pos * 0.5 + 0.5) / zoom - offset;
}
