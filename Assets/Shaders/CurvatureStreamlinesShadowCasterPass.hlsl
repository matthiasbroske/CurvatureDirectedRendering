#ifndef CURVATURE_STREAMLINES_SHADOW_CASTER_PASS_INCLUDED
#define CURVATURE_STREAMLINES_SHADOW_CASTER_PASS_INCLUDED

// ===== Includes =====
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
#include "Assets/Compute/DrawIndirectHelpers.hlsl"

// ===== Structs =====
struct Attributes
{
    uint vertexID : SV_VERTEXID;
};

struct Interpolators
{
    float4 positionCS : SV_POSITION;
};

// ===== Uniforms =====
StructuredBuffer<DrawTriangle> _DrawTriangles;
float3 _LightDirection;
float4x4 _ObjectToWorld;

// ===== Helpers =====
float3 FlipNormalBasedOnViewDir(float3 normalWS, float3 positionWS)
{
    float3 viewDirWS = GetWorldSpaceNormalizeViewDir(positionWS);
    return normalWS * (dot(normalWS, viewDirWS) < 0 ? - 1 : 1);
}

float4 GetShadowCasterPositionCS(float3 positionWS, float3 normalWS)
{
    float3 lightDirectionWS = _LightDirection;
    normalWS = FlipNormalBasedOnViewDir(normalWS, positionWS);
    
    float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, lightDirectionWS));
    #if UNITY_REVERSED_Z
        positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
    #else
        positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
    #endif
    return positionCS;
}

// ===== Vert =====
Interpolators Vertex(Attributes input)
{
    Interpolators output = (Interpolators)0;

    DrawTriangle tri = _DrawTriangles[input.vertexID / 3];
    DrawVertex vert = tri.vertices[input.vertexID % 3];

    float3 positionWS = mul(_ObjectToWorld, float4(vert.positionOS, 1.0)).xyz;
    float3 normalWS = mul(vert.normalOS, (float3x3)_ObjectToWorld);

    output.positionCS = GetShadowCasterPositionCS(positionWS, normalWS);

    return output;
}

// ===== Frag =====
half4 Fragment(Interpolators input) : SV_TARGET
{
    return 0;
}

#endif