#version 460

// Heightmap data — zero-copy from compute shader's GpuBuffer
layout(set = 0, binding = 1) readonly buffer HeightmapData {
    float heights[];
};

// Per-frame uniform buffer
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

// Grid dimensions
layout(push_constant) uniform PushConstants {
    int   grid_width;
    int   grid_height;
    float hmap_w;       // width of the terrain quad
    uint  flags;        // bit 0: add_skirt, bit 1: wireframe
};

layout(location = 0) out vec3 frag_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out vec4 frag_pos_light_space;

void main()
{
    int vertex_id = gl_VertexIndex;
    int total_grid = grid_width * grid_height;

    // Decode grid coordinates from vertex index
    int x, z;
    float y;

    if (vertex_id < total_grid)
    {
        // Main grid vertex
        x = vertex_id % grid_width;
        z = vertex_id / grid_width;
        y = heights[z * grid_width + x];
    }
    else
    {
        // Skirt vertex (below the grid edge) — extend downward
        int skirt_id = vertex_id - total_grid;
        int perimeter = 2 * (grid_width + grid_height) - 4;
        int edge_id = skirt_id % perimeter;

        // Walk around the perimeter: bottom, right, top, left
        if (edge_id < grid_width)
        {
            x = edge_id; z = 0;
        }
        else if (edge_id < grid_width + grid_height - 1)
        {
            x = grid_width - 1; z = edge_id - grid_width + 1;
        }
        else if (edge_id < 2 * grid_width + grid_height - 2)
        {
            x = grid_width - 1 - (edge_id - grid_width - grid_height + 2); z = grid_height - 1;
        }
        else
        {
            x = 0; z = grid_height - 1 - (edge_id - 2 * grid_width - grid_height + 3);
        }

        // Alternate between edge height and skirt floor
        if (skirt_id >= perimeter)
            y = -0.05; // skirt bottom
        else
            y = heights[z * grid_width + x];
    }

    // Map to world coordinates: centered at origin, width = hmap_w
    float fx = float(x) / float(grid_width - 1);
    float fz = float(z) / float(grid_height - 1);

    vec3 pos = vec3(
        (fx - 0.5) * hmap_w,
        (y + hmap_h0) * hmap_h * scale_h,
        (fz - 0.5) * hmap_w
    );

    // Compute normal from finite differences
    float hL = (x > 0)              ? heights[z * grid_width + (x - 1)] : y;
    float hR = (x < grid_width - 1) ? heights[z * grid_width + (x + 1)] : y;
    float hD = (z > 0)              ? heights[(z - 1) * grid_width + x] : y;
    float hU = (z < grid_height - 1)? heights[(z + 1) * grid_width + x] : y;

    float dx = hmap_w / float(grid_width - 1);
    float dz = hmap_w / float(grid_height - 1);
    vec3 normal = normalize(vec3(
        (hL - hR) * hmap_h * scale_h / (2.0 * dx),
        1.0,
        (hD - hU) * hmap_h * scale_h / (2.0 * dz)
    ));

    // UV for texture sampling
    frag_uv = vec2(fx, fz);

    // Transform
    vec4 world_pos = model * vec4(pos, 1.0);
    frag_pos = world_pos.xyz;
    frag_normal = mat3(transpose(inverse(model))) * normal;
    frag_pos_light_space = light_space_matrix * world_pos;

    gl_Position = projection * view * world_pos;
}
