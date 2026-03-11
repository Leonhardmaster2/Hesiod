#version 460

layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 frag_pos_light_space;

layout(location = 0) out vec4 out_color;

// Per-frame uniform buffer (same as vertex shader)
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

// Material / rendering params
layout(set = 0, binding = 2) uniform MaterialUBO {
    vec3  base_color;
    float gamma_correction;
    float shadow_strength;
    float spec_strength;
    float shininess;
    float normal_map_scaling;
    uint  material_flags;  // bit 0: use_texture_albedo, bit 1: bypass_shadow, bit 2: normal_vis
    float ao_strength;
    float ao_radius;
    float padding0;
};

// Textures
layout(set = 1, binding = 0) uniform sampler2D texture_albedo;
layout(set = 1, binding = 1) uniform sampler2D texture_hmap;
layout(set = 1, binding = 2) uniform sampler2D texture_normal;
layout(set = 1, binding = 3) uniform sampler2D texture_shadow_map;

// Decode material flags
bool use_texture_albedo() { return (material_flags & 1u) != 0u; }
bool bypass_shadow_map()  { return (material_flags & 2u) != 0u; }
bool normal_visualization() { return (material_flags & 4u) != 0u; }
bool add_ao()             { return (material_flags & 8u) != 0u; }

// Shadow calculation with PCF
float calculate_shadow(vec4 pos_light_space, vec3 light_dir, vec3 norm)
{
    vec3 proj = pos_light_space.xyz / pos_light_space.w;
    proj = proj * 0.5 + 0.5;

    // Vulkan NDC: Y is flipped compared to OpenGL
    proj.y = 1.0 - proj.y;

    if (proj.z > 1.0) return 0.0;

    float current_depth = proj.z;
    float bias_t = clamp(dot(norm, light_dir), 0.0, 1.0);
    float bias = mix(1e-4, 1e-4, bias_t);

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(texture_shadow_map, 0);
    float sum = 0.0;
    int ir = 2;

    for (int x = -ir; x <= ir; ++x)
    {
        for (int y = -ir; y <= ir; ++y)
        {
            float pcf_depth = texture(texture_shadow_map, proj.xy + vec2(x, y) * texel_size).r;
            float weight = 1.0 - length(vec2(x, y)) / float(ir + 1);
            shadow += weight * (current_depth - bias > pcf_depth ? 1.0 : 0.0);
            sum += weight;
        }
    }
    return shadow / sum;
}

// Horizon-based AO
float gain(float x, float factor)
{
    return x < 0.5 ? 0.5 * pow(2.0 * x, factor)
                   : 1.0 - 0.5 * pow(2.0 * (1.0 - x), factor);
}

float compute_hbao(vec2 uv_coord, int dir_count, int step_count, float radius)
{
    float h0 = texture(texture_hmap, uv_coord).r;
    float occlusion = 0.0;

    for (int d = 0; d < dir_count; d++)
    {
        float dir_angle = 6.28318530718 * float(d) / float(dir_count);
        vec2 dir = vec2(cos(dir_angle), sin(dir_angle));
        float horizon = -1.5707963;
        float sc = 2.0;

        for (int s = 1; s <= step_count; s++)
        {
            float lf = float(s) / float(step_count);
            float t = (exp2(lf) - 1.0);
            vec2 suv = clamp(uv_coord + dir * t * radius, 0.0, 1.0);
            float hs = texture(texture_hmap, suv).r;
            float slope = sc * (hs - h0) / (t * radius);
            horizon = max(horizon, atan(slope));
        }
        occlusion += max(0.0, horizon);
    }

    float ao = 1.0 - (occlusion / float(dir_count)) / 1.5707963;
    ao = clamp(ao, 0.0, 1.0);
    ao = gain(ao, 3.0);
    ao = 1.0 - ao_strength + ao * ao_strength;
    return ao;
}

void main()
{
    vec3 normal = normalize(frag_normal);

    // Normal map detail
    if (normal_map_scaling > 0.0)
    {
        vec3 nd = texture(texture_normal, frag_uv).xyz;
        nd = vec3(nd.x, nd.z, nd.y);
        normal = normalize(normal + normal_map_scaling * nd);
    }

    // Normal visualization mode
    if (normal_visualization())
    {
        out_color = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }

    // Base color
    vec3 color;
    if (use_texture_albedo())
        color = texture(texture_albedo, frag_uv).xyz;
    else
        color = base_color;

    // Gamma decode
    color = pow(color, vec3(1.0 / gamma_correction));

    // Lighting vectors
    vec3 norm = normal;
    vec3 light_dir = normalize(light_pos - frag_pos);
    vec3 view_dir = normalize(view_pos - frag_pos);

    // Diffuse
    float diff = max(dot(norm, light_dir), 0.0);

    // Specular (Blinn-Phong)
    vec3 half_dir = normalize(light_dir + view_dir);
    float spec = spec_strength * pow(max(dot(norm, half_dir), 0.0), shininess);

    // Shadow
    float shadow = 0.0;
    if (!bypass_shadow_map())
        shadow = calculate_shadow(frag_pos_light_space, light_dir, norm);

    spec *= (1.0 - shadow);

    // Combine lighting
    float diff_m = min(diff, 1.0 - shadow);
    diff_m = 1.0 - shadow_strength + shadow_strength * smoothstep(1.0 - shadow_strength, 1.0, diff_m);

    vec3 diffuse = color * diff_m;
    vec3 specular = spec * vec3(1.0);
    vec3 ambient = 0.2 * color;
    vec3 result = ambient + diffuse + specular;

    // AO
    if (add_ao())
    {
        float ao = compute_hbao(frag_uv, 16, 8, ao_radius);
        result *= pow(ao, 0.5);
    }

    out_color = vec4(result, 1.0);
}
