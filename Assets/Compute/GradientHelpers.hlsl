#ifndef GRADIENT_HELPERS_INCLUDED
#define GRADIENT_HELPERS_INCLUDED

#include "Packages/com.matthias.utilities/Runtime/Compute/VoxelHelpers.hlsl"

////////////////////////////////////////////////////////////////////////////////
///                                Uniforms                                  ///
////////////////////////////////////////////////////////////////////////////////
RWStructuredBuffer<float3> _Gradients;

////////////////////////////////////////////////////////////////////////////////
///                                Methods                                   ///
////////////////////////////////////////////////////////////////////////////////
float3 GradientValue(uint x, uint y, uint z)
{
    return _Gradients[VoxelIdx(x, y, z)];
}
float3 GradientValue(uint3 voxelId)
{
    return GradientValue(voxelId.x, voxelId.y, voxelId.z);
}
float3 GradientValueTrilinear(float3 uvw) {
    return Float3ValueTrilinear(uvw, _Gradients);
}
float4 VoxelGradientWithValue(uint3 i)
{
    return float4(GradientValue(i), VoxelValue(i));
}

#endif