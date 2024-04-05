// Created with help from the following excellent URP shader tutorial:
// https://nedmakesgames.github.io/blog/urp-hlsl-shaders
Shader "Curvature/Streamlines"
{
    Properties
    {
        _FrontColor ("Front Color", Color) = (1, 1, 1, 1)
        _BackColor ("Back Color", Color) = (0.5, 0.5, 0.5, 1)
        _Smoothness ("Smoothness", Range(0, 1)) = 0.5
        _Metalness ("Metalness", Range(0, 1)) = 0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "IgnoreProjector"="True"}

        Pass
        {
            Name "ForwardLit"
            Tags {"LightMode" = "UniversalForward"}
            
            Cull Off

            HLSLPROGRAM
            
            // Signal this shader requires a compute buffer
            #pragma prefer_hlslcc gles
            #pragma exclude_renderers d3d11_9x
            #pragma target 5.0

            // Lighting and shadow keywords
            #define _SPECULAR_COLOR
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile_fragment _ _SHADOWS_SOFT

            // Vert/frag declarations
            #pragma vertex Vertex
            #pragma fragment Fragment
            
            #include "CurvatureStreamlinesForwardLitPass.hlsl"

            ENDHLSL
        }
        
        Pass
        {
            Name "ShadowCaster"
            Tags {"LightMode" = "ShadowCaster"}
            
            ColorMask 0  // No need to write color during a shadow caster pass
            
            HLSLPROGRAM

            // Vert/frag declarations
            #pragma vertex Vertex
            #pragma fragment Fragment

            #include "CurvatureStreamlinesShadowCasterPass.hlsl"

            ENDHLSL
        }
    }
}
