#ifndef CURVATURE_STREAMLINES_INCLUDED
#define CURVATURE_STREAMLINES_INCLUDED

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
    float3 positionWS : TEXCOORD1;
    float3 normalWS : TEXCOORD2;
};

// ===== Uniforms =====
StructuredBuffer<DrawTriangle> _DrawTriangles;
float4 _FrontColor;
float4 _BackColor;
float _Smoothness;
float _Metalness;
float4x4 _ObjectToWorld;
float4x4 _WorldToObject;

// ===== Vert =====
Interpolators Vertex(Attributes input)
{
    Interpolators output = (Interpolators)0;

    DrawTriangle tri = _DrawTriangles[input.vertexID / 3];
    DrawVertex vert = tri.vertices[input.vertexID % 3];

    output.positionWS = mul(_ObjectToWorld, float4(vert.positionOS, 1.0)).xyz;
    output.normalWS = mul(vert.normalOS, (float3x3)_WorldToObject);
    output.positionCS = TransformWorldToHClip(output.positionWS);

    return output;
}

// ===== Frag =====
half4 Fragment(Interpolators input, bool isFrontFace: SV_IsFrontFace) : SV_TARGET
{
    InputData lightingInput = (InputData)0;
    lightingInput.positionWS = input.positionWS;
    lightingInput.normalWS = normalize(isFrontFace ? input.normalWS : -input.normalWS);  // Flip normal for back faces
    lightingInput.viewDirectionWS = GetWorldSpaceNormalizeViewDir(input.positionWS);
    lightingInput.shadowCoord = TransformWorldToShadowCoord(input.positionWS);
    lightingInput.vertexLighting = 0;
    
    SurfaceData surfaceData = (SurfaceData)0;
    surfaceData.alpha = 1;
    surfaceData.albedo = isFrontFace ? _FrontColor.rgb : _BackColor.rgb;
    surfaceData.specular = 1;
    surfaceData.smoothness = _Smoothness;
    surfaceData.metallic = _Metalness;
    
    return UniversalFragmentPBR(lightingInput, surfaceData) + _GlossyEnvironmentColor;  // Hack to include ambient environment color since reflection is broken for instanced rendering
}

#endif