# CurvatureDirectedRendering
<p align="center">
  <img alt="Demo Scene" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/57412d42-ad15-484c-9e9d-a70950025342">
</p>

## About
Curvature-directed lines for conveying the shape of 3D SDF surfaces in a perceptually optimal manner, modernized to run in real time on the GPU. 

### Features
- Principal direction/curvature derivations for SDFs
- Parallel Poisson disk point sampling across SDF surfaces on the GPU
- Procedural geometry generation with compute shaders

## Demo
A demonstration of this technique applied to the Stanford Bunny is available in the `Demo` scene in the `Assets/Scenes` folder, which can be accessed by downloading this repository and opening with Unity 2022.3 or later.

To experiment with the demo, press play in the `Demo` scene and adjust the various sliders in the top left corner to alter the visualization in real time.

There are three additional toggles in the top right corner that are meant to be left on, but can be toggled off to demonstrate how their associated features are necessary in maximizing the perceptual effectiveness of the technique. They are as follows:
- **Taper:** Whether or not to taper the lines. Creates softer, more visually appealing lines.
- **Scale by Curvature:** Whether or not to scale the length and width of lines by the value of first principal curvature. Essential for conveying the shape of the object.
- **Poisson Disk Sampling:** Whether or not to use poisson disk sampling to generate the initial starting points for the lines. Necessary to avoid overlapping lines and inconsistent empty space.

## Usage
This is a general technique that can be applied to the rendering of any 3D SDF surface. To apply this technique to an SDF of your own, simply use the [`CurvatureRenderer.cs`](Assets/Scripts/CurvatureRenderer.cs) class as demonstrated in the `Demo` scene.

## Credit
The technique presented here is a modernization of Victoria Interrante's excellent research on illustrating surface shape with curvature:
- *[Illustrating surface shape in volume data via principal direction-driven 3D line integral convolution](https://doi.org/10.1145/258734.258796)*
- *[Illustrating transparent surfaces with curvature-directed strokes](https://doi.org/10.1109/VISUAL.1996.568110)*
- *[Conveying the 3D shape of smoothly curving transparent surfaces via texture](https://doi.org/10.1109/2945.597794)*

The GPU poisson disk sampling algorithm used is an interpretation of the technique described in the following paper:
- *[Parallel Poisson disk sampling with spectrum analysis on surfaces](https://dl.acm.org/doi/10.1145/1882261.1866188)*

The SDF for the Stanford Bunny was generated using the following python package:
- [mesh-to-sdf](https://pypi.org/project/mesh-to-sdf/)
