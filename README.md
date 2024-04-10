# CurvatureDirectedRendering
<p align="center">
  <img alt="Demo Scene" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/b0218719-855b-455d-9412-36dd23e7bb35">
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
<p align="left">
  <img alt="Taper" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/84556c98-fabb-4108-be39-f160218d862c" width="40%">
&nbsp; &nbsp;
  <img alt="No Taper" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/f5840afa-ddd0-478a-86b6-7dd1a356f523" width="40%">
</p>

- **Scale by Curvature:** Whether or not to scale the length and width of lines by the value of first principal curvature. Essential for conveying the shape of the object.
<p align="left">
  <img alt="Scale by Curvature" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/9e951a6b-35b7-490e-a06a-02dc47d9859b" width="40%">
&nbsp; &nbsp;
  <img alt="Constant Scale" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/ed44b1af-034a-4c4d-84bf-476bb3eaba3b" width="40%">
</p>

- **Poisson Disk Sampling:** Whether or not to use poisson disk sampling to generate the initial starting points for the lines. Necessary to avoid overlapping lines and inconsistent empty space.
<p align="left">
  <img alt="Poisson Disk Sampling" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/adbd709e-3bdb-4eb7-8daf-9b08dedc03c3" width="40%">
&nbsp; &nbsp;
  <img alt="Random Sampling" src="https://github.com/matthiasbroske/CurvatureDirectedRendering/assets/82914350/f673ae17-e913-44aa-a6b8-d4068549ebd9" width="40%">
</p>

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
