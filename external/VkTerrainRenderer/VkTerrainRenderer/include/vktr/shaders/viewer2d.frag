#version 460

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D texture_hmap;
layout(set = 0, binding = 1) uniform sampler2D texture_normal;

layout(push_constant) uniform PushConstants {
    float aspect_ratio;
    float zoom;
    vec2  offset;
    float sun_azimuth;
    float sun_zenith;
    int   cmap;           // 0=gray, 1=viridis, 2=turbo, 3=magma
    uint  flags;          // bit 0: hillshading
};

// Colormap functions
vec3 turbo(float t)
{
    const vec3 c0 = vec3(0.1140890109226559, 0.06288340699912215, 0.2248337216805064);
    const vec3 c1 = vec3(6.716419496985708, 3.182286745507602, 7.571581586103393);
    const vec3 c2 = vec3(-66.09402360453038, -4.9279827041226, -10.09439367561635);
    const vec3 c3 = vec3(228.7660791526501, 25.04986699771073, -91.54105330182436);
    const vec3 c4 = vec3(-334.8351565777451, -69.31749712757485, 288.5858850615712);
    const vec3 c5 = vec3(218.7637218434795, 67.52150567819112, -305.2045772184957);
    const vec3 c6 = vec3(-52.88903478218835, -21.54527364654712, 110.5174647748972);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 viridis(float t)
{
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 magma(float t)
{
    const vec3 c0 = vec3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
    const vec3 c1 = vec3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
    const vec3 c2 = vec3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
    const vec3 c3 = vec3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
    const vec3 c4 = vec3(52.17613981234068, -27.94360607168351, 12.94416944238394);
    const vec3 c5 = vec3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
    const vec3 c6 = vec3(18.65570506591883, -11.48977351997711, -5.601961508734096);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

float hillshade(vec3 n)
{
    float slope = acos(n.z);
    float aspect = atan(n.y, n.x);
    float sh = cos(sun_zenith) * cos(slope) +
               sin(sun_zenith) * sin(slope) * cos(sun_azimuth - aspect);
    return clamp(sh, 0.0, 1.0);
}

void main()
{
    if (frag_uv.x < 0.0 || frag_uv.x > 1.0 || frag_uv.y < 0.0 || frag_uv.y > 1.0)
    {
        out_color = vec4(0.1, 0.1, 0.1, 1.0);
        return;
    }

    float h = texture(texture_hmap, frag_uv).r;

    vec3 color;
    if (cmap == 0)
        color = vec3(h);
    else if (cmap == 1)
        color = viridis(h);
    else if (cmap == 2)
        color = turbo(h);
    else
        color = magma(h);

    if ((flags & 1u) != 0u) // hillshading
    {
        vec3 n = texture(texture_normal, frag_uv).xyz;
        n = vec3(n.x, n.z, n.y); // swizzle to match coordinate system
        color *= hillshade(n);
    }

    out_color = vec4(color, 1.0);
}
