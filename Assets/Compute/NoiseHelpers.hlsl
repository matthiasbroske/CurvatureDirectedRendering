#ifndef NOISE_HELPERS_INCLUDED
#define NOISE_HELPERS_INCLUDED

#include "Packages/com.matthias.utilities/Runtime/Compute/VoxelHelpers.hlsl"

RWStructuredBuffer<float> _WhiteNoise;

float Hash13(float3 p3)
{
    p3  = frac(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

float WhiteNoiseValue(uint3 i) {
    return _WhiteNoise[VoxelIdx(i)];
}

float WhiteNoiseValueTrilinear(float3 uvw) {
    return FloatValueTrilinear(uvw, _WhiteNoise);
}

#endif