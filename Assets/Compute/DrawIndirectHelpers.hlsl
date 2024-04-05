#ifndef DRAW_INDIRECT_HELPERS_INCLUDED
#define DRAW_INDIRECT_HELPERS_INCLUDED

////////////////////////////////////////////////////////////////////////////////
///                                 Structs                                  ///
////////////////////////////////////////////////////////////////////////////////
struct DrawVertex
{
    float3 positionOS;
    float3 normalOS;
};

struct DrawTriangle {
    DrawVertex vertices[3];
};

struct IndirectArgs
{
    uint numVerticesPerInstance;
    uint numInstance;
    uint startVertexIndex;
    uint startInstanceIndex;
};

////////////////////////////////////////////////////////////////////////////////
///                                Uniforms                                  ///
////////////////////////////////////////////////////////////////////////////////
RWStructuredBuffer<IndirectArgs> _IndirectArgs;

#endif