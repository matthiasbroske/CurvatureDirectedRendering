#ifndef DISPATCH_INDIRECT_BUFFERS_INCLUDED
#define DISPATCH_INDIRECT_BUFFERS_INCLUDED

////////////////////////////////////////////////////////////////////////////////
///                                Uniforms                                  ///
////////////////////////////////////////////////////////////////////////////////
uint _ThreadCount;
RWStructuredBuffer<int> _PointCounterBuffer;
RWStructuredBuffer<int> _ThreadGroupCounterBuffer;

#endif