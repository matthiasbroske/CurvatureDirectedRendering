#ifndef MARCHING_CUBES_HELPERS_INCLUDED
#define MARCHING_CUBES_HELPERS_INCLUDED

////////////////////////////////////////////////////////////////////////////////
/// Params
////////////////////////////////////////////////////////////////////////////////
float _IsoValue;

////////////////////////////////////////////////////////////////////////////////
/// Buffers
////////////////////////////////////////////////////////////////////////////////
StructuredBuffer<uint2> _TriangleTable;

////////////////////////////////////////////////////////////////////////////////
/// Helper Methods
////////////////////////////////////////////////////////////////////////////////
uint EdgeIndexFromTriangleTable(uint2 data, uint index)
{
    return 0xfu & (index < 8 ? data.x >> ((index + 0) * 4) :
                               data.y >> ((index - 8) * 4));
}

uint2 EdgeVertexPair(uint index)
{
    // (0, 1) (1, 2) (2, 3) (3, 0)
    // (4, 5) (5, 6) (6, 7) (7, 4)
    // (0, 4) (1, 5) (2, 6) (3, 7)
    uint v1 = index & 7;
    uint v2 = index < 8 ? ((index + 1) & 3) | (index & 4) : v1 + 4;
    return uint2(v1, v2);
}

uint3 CubeVertex(uint index)
{
    bool x = index & 1;
    bool y = index & 2;
    bool z = index & 4;
    return uint3(x ^ y, y, z);
}

#endif