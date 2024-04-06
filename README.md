# CurvatureDirectedRendering

## About
Curvature-directed lines for conveying the shape of 3D SDF surfaces in a perceptually optimal manner, modernized to run in real time on the GPU. 

### Features
- Principal direction/curvature derivations for SDFs
- Parallel Poisson disk point sampling across SDF surfaces on the GPU
- Procedural streamline geometry generation with compute shaders

## Usage
Download this repository, open with Unity 2022.3 or later, and proceed to the `Demo` scene in the `Assets/Scenes` folder. Press play and adjust the various sliders to alter the visualization in real time.

## Credit
This work is a direct modernization of Victoria Interrante's excellent research on illustrating surface shape with curvature:
- *[Illustrating surface shape in volume data via principal direction-driven 3D line integral convolution](https://doi.org/10.1145/258734.258796)*
- *[Illustrating transparent surfaces with curvature-directed strokes](https://doi.org/10.1109/VISUAL.1996.568110)*
- *[Conveying the 3D shape of smoothly curving transparent surfaces via texture](https://doi.org/10.1109/2945.597794)*

The GPU poisson disk sampling algorithm used is an interpretation of the technique described in the following paper:
- *[Parallel Poisson disk sampling with spectrum analysis on surfaces](https://dl.acm.org/doi/10.1145/1882261.1866188)*

The SDF for the Stanford Bunny was generated using the following python package:
- [mesh-to-sdf](https://pypi.org/project/mesh-to-sdf/)
