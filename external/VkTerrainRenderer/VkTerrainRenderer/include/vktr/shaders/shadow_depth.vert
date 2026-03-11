#version 460

// Heightmap data — same SSBO as terrain.vert
layout(set = 0, binding = 1) readonly buffer HeightmapData {
    float heights[];
};

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4  model;
    mat4  view;
    mat4  projection;
    mat4  light_space_matrix;
    vec3  light_pos;
    float scale_h;
    vec3  camera_pos;
    float hmap_h;
    vec3  view_pos;
    float hmap_h0;
};

layout(push_constant) uniform PushConstants {
    int   grid_width;
    int   grid_height;
    float hmap_w;
    uint  flags;
};

void main()
{
    int vertex_id = gl_VertexIndex;
    int x = vertex_id % grid_width;
    int z = vertex_id / grid_width;

    // Clamp to grid bounds for safety
    x = min(x, grid_width - 1);
    z = min(z, grid_height - 1);

    float y = heights[z * grid_width + x];

    float fx = float(x) / float(grid_width - 1);
    float fz = float(z) / float(grid_height - 1);

    vec3 pos = vec3(
        (fx - 0.5) * hmap_w,
        (y + hmap_h0) * hmap_h * scale_h,
        (fz - 0.5) * hmap_w
    );

    gl_Position = light_space_matrix * model * vec4(pos, 1.0);
}
