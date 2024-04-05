#ifndef PRINCIPAL_CURVATURE_HELPERS_INCLUDED
#define PRINCIPAL_CURVATURE_HELPERS_INCLUDED

#include "Packages/com.matthias.utilities/Runtime/Compute/VoxelHelpers.hlsl"

////////////////////////////////////////////////////////////////////////////////
///                                Uniforms                                  ///
////////////////////////////////////////////////////////////////////////////////
RWStructuredBuffer<float4> _PrincipalCurvatures;
float _CurvatureScale;

////////////////////////////////////////////////////////////////////////////////
///                                Methods                                   ///
////////////////////////////////////////////////////////////////////////////////
float4 PrincipalCurvatureValue(uint x, uint y, uint z)
{
    return _PrincipalCurvatures[VoxelIdx(x, y, z)];
}
float4 PrincipalCurvatureValue(uint3 i)
{
    return PrincipalCurvatureValue(i.x, i.y, i.z);
}

float4 PrincipalCurvatureValueTrilinear(float3 uvw)
{
    return Float4ValueTrilinear(uvw, _PrincipalCurvatures);
}

float3 PrincipalCurvatureDirectionTrilinear(float3 uvw)
{
    return PrincipalCurvatureValueTrilinear(uvw).xyz;
}

float3 RK4CurvatureDirectionFromPosition(float3 position, float h, float3 guideDirection)
{
    float3 uvw = RemapPositionToUVW(position);
    float3 k1 = normalize(PrincipalCurvatureDirectionTrilinear(uvw));
    k1 *= (dot(guideDirection, k1) > 0) ? 1 : -1;

    uvw = RemapPositionToUVW(position+h/2*k1);
    float3 k2 = normalize(PrincipalCurvatureDirectionTrilinear(uvw));
    k2 *= (dot(guideDirection, k2) > 0) ? 1 : -1;

    uvw = RemapPositionToUVW(position+h/2*k2);
    float3 k3 = normalize(PrincipalCurvatureDirectionTrilinear(uvw));
    k3 *= (dot(guideDirection, k3) > 0) ? 1 : -1;

    uvw = RemapPositionToUVW(position+h*k3);
    float3 k4 = normalize(PrincipalCurvatureDirectionTrilinear(uvw+h*k3));
    k4 *= (dot(guideDirection, k4) > 0) ? 1 : -1;

    return h/6*(k1 + 2*k2 + 2*k3 + 1*k4);
}

#endif