using UnityEngine;
using Matthias.Utilities;
using UnityEngine.Rendering;

namespace Curvature
{
    /// <summary>
    /// Renders curvature directed streamlines by dispatching compute shaders on the GPU.
    /// </summary>
    public class CurvatureRenderer : MonoBehaviour
    {
        [Header("Render Settings")]
        [SerializeField] private ShadowCastingMode _shadowCastingMode = ShadowCastingMode.TwoSided;
        [Header("Dependencies - Compute")]
        [SerializeField] private ComputeShader _gradientCompute;
        [SerializeField] private ComputeShader _surfaceSamplerCompute;
        [SerializeField] private ComputeShader _poissonCompute;
        [SerializeField] private ComputeShader _principalCurvatureCompute;
        [SerializeField] private ComputeShader _licCompute;
        [SerializeField] private ComputeShader _marchingCubesCompute;
        [SerializeField] private ComputeShader _streamlineBuilderCompute;
        [Header("Dependencies - Material")]
        [SerializeField] private Material _curvatureStreamlinesMaterial;
        
        // Stride lengths
        private const int DRAW_INDIRECT_ARGS_STRIDE = sizeof(int) * 4;
        private const int DISPATCH_INDIRECT_ARGS_STRIDE = sizeof(int) * 3;
        private const int DRAW_TRIANGLE_STRIDE = sizeof(float) * 3 * (3 + 3);  // 3 * (position + normal)
        private const int VOXEL_STRIDE = sizeof(float);
        private const int MAX_POISSON_POINTS_PER_VOXEL = 4;
        private const int POINT_STRIDE = sizeof(float) * 3;
        private const int MIN_MAX_STRIDE = sizeof(int) * 2;

        // SDF
        private SDF _sdf;

        // Compute buffers
        private ComputeBuffer _streamlineTriangleBuffer;
        private ComputeBuffer _voxelBuffer;
        private ComputeBuffer _drawArgsBuffer;
        private ComputeBuffer _poissonDispatchArgsBuffer;
        private ComputeBuffer _streamlineDispatchArgsBuffer;
        private ComputeBuffer _gradientBuffer;
        private ComputeBuffer _surfacePointsBuffer;
        private ComputeBuffer _surfacePointsCountBuffer;
        private ComputeBuffer _pointCounterBuffer;
        private ComputeBuffer _groupCounterBuffer;
        private ComputeBuffer _cellsBuffer;
        private ComputeBuffer _pointsByCellBuffer;
        private ComputeBuffer _poissonBuffer;
        private ComputeBuffer _poissonCountBuffer;
        private ComputeBuffer _principalCurvatureBuffer;
        private ComputeBuffer _curvatureMinMaxBuffer;
        private ComputeBuffer _noiseBuffer;
        private ComputeBuffer _licBuffer;
        private ComputeBuffer _triangleTableBuffer;
        // Buffer resets
        private readonly int[] _drawArgsBufferReset = { 0, 1, 0, 0 };
        private readonly int[] _dispatchArgsBufferReset = { 1, 1, 1 };
        private readonly int[] _minMaxBufferReset = { int.MaxValue, 0 };
        // Compute kernels
        private int _gradientKernel;
        private int _sampleSurfaceKernel;
        private int _poissonInitKernel;
        private int _poissonSelectKernel;
        private int _poissonRemoveKernel;
        private int _poissonCollapseKernel;
        private int _curvatureKernel;
        private int _minMaxCurvatureKernel;
        private int _streamlineKernel;
        private int _noiseKernel;
        private int _particlesKernel;
        private int _licKernel;
        private int _marchKernel;
        // Demonstration purposes only
        private bool _usePoisson = true;
        // Thread groups
        private Vector3Int _gradientThreadGroups;
        private Vector3Int _sampleSurfaceThreadGroups;
        private uint _poissonInitThreadGroupsX, _poissonInitThreadGroupsY, _poissonInitThreadGroupsZ;
        private uint _poissonSelectThreadGroupsX;
        private uint _poissonRemoveThreadGroupsX, _poissonRemoveThreadGroupsY, _poissonRemoveThreadGroupsZ;
        private uint _poissonCollapseThreadGroupsX, _poissonCollapseThreadGroupsY, _poissonCollapseThreadGroupsZ;
        private Vector3Int _curvatureThreadGroups;
        private uint _streamlineThreadGroupsX;
        private Vector3Int _noiseThreadGroups;
        private Vector3Int _licThreadGroups;
        private Vector3Int _mcThreadGroups;
        
        // Poisson
        private float _poissonRadius;

        // Initialized
        private bool _initialized = false;

        /// <summary>
        /// Initializes the marching cubes renderer with a volume to render.
        /// </summary>
        public void Init(SDF sdf, float curvatureScale, float length, float width, float spacing)
        {
            _sdf = sdf;
            
            // Set default parameters
            _principalCurvatureCompute.SetFloat("_CurvatureScale", curvatureScale);
            _streamlineBuilderCompute.SetFloat("_MaxLength", length);
            _streamlineBuilderCompute.SetFloat("_Length", length / 5);
            _streamlineBuilderCompute.SetFloat("_MaxWidth", width);
            _streamlineBuilderCompute.SetFloat("_Width", width / 5);
            _poissonRadius = Mathf.Max(_sdf.VoxelSpacing.x, spacing);
            
            // Get kernels
            _gradientKernel = _gradientCompute.FindKernel("Gradient");
            _sampleSurfaceKernel = _surfaceSamplerCompute.FindKernel("SampleSurfacePoints");
            _poissonInitKernel = _poissonCompute.FindKernel("Initialize");
            _poissonSelectKernel = _poissonCompute.FindKernel("SelectPoints");
            _poissonRemoveKernel = _poissonCompute.FindKernel("RemovePoints");
            _poissonCollapseKernel = _poissonCompute.FindKernel("CollapsePoints");
            _curvatureKernel = _principalCurvatureCompute.FindKernel("PrincipalCurvature");
            _minMaxCurvatureKernel = _streamlineBuilderCompute.FindKernel("MinMaxCurvature");
            _streamlineKernel = _streamlineBuilderCompute.FindKernel("BuildStreamline");
            _noiseKernel = _licCompute.FindKernel("InitNoise");
            _particlesKernel = _licCompute.FindKernel("InitParticles");
            _licKernel = _licCompute.FindKernel("LIC");
            _marchKernel = _marchingCubesCompute.FindKernel("March");
            
            // Get thread groups
            _gradientThreadGroups = ComputeUtilities.GetThreadGroups(_gradientCompute, _gradientKernel, _sdf.Dimensions);
            _sampleSurfaceThreadGroups = ComputeUtilities.GetThreadGroups(_surfaceSamplerCompute, _sampleSurfaceKernel, _sdf.Dimensions);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonInitKernel, out _poissonInitThreadGroupsX, out _poissonInitThreadGroupsY, out _poissonInitThreadGroupsZ);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonSelectKernel, out _poissonSelectThreadGroupsX, out _, out _);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonRemoveKernel, out _poissonRemoveThreadGroupsX, out _poissonRemoveThreadGroupsY, out _poissonRemoveThreadGroupsZ);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonCollapseKernel, out _poissonCollapseThreadGroupsX, out _poissonCollapseThreadGroupsY, out _poissonCollapseThreadGroupsZ);
            _curvatureThreadGroups = ComputeUtilities.GetThreadGroups(_principalCurvatureCompute, _curvatureKernel, _sdf.Dimensions);
            _streamlineBuilderCompute.GetKernelThreadGroupSizes(_streamlineKernel, out _streamlineThreadGroupsX, out _, out _);
            _noiseThreadGroups = ComputeUtilities.GetThreadGroups(_licCompute, _noiseKernel, _sdf.Dimensions);
            _licThreadGroups = ComputeUtilities.GetThreadGroups(_licCompute, _licKernel, _sdf.Dimensions);
            _mcThreadGroups = ComputeUtilities.GetThreadGroups(_marchingCubesCompute, _marchKernel, _sdf.Dimensions);
            
            // Flag as initialized
            _initialized = true;
        }

        /// <summary>
        /// Renders the curvature-directed streamlines, executing the entire rendering pipeline.
        /// </summary>
        public void Render()
        {
            if (!_initialized) return;
            
            // Initialize the compute buffers
            CreateBuffers();
            SetBuffers();
            
            // Initialize dispatch args buffers
            _poissonDispatchArgsBuffer.SetData(_dispatchArgsBufferReset);
            _streamlineDispatchArgsBuffer.SetData(_dispatchArgsBufferReset);

            // Initialize voxel buffer with SDF values
            _voxelBuffer.SetData(_sdf.Voxels);
            
            // Set surface sampler compute uniforms
            _surfaceSamplerCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            _surfaceSamplerCompute.SetVector("_VoxelStartPosition", _sdf.StartPosition);
            _surfaceSamplerCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _surfaceSamplerCompute.SetInt("_ThreadCount", (int)_poissonSelectThreadGroupsX);
            
            // Set poisson compute uniforms
            _poissonCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            _poissonCompute.SetVector("_VoxelStartPosition", _sdf.StartPosition);
            _poissonCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _poissonCompute.SetInt("_ThreadCount", (int)_streamlineThreadGroupsX);
            
            // Set gradient compute uniforms
            _gradientCompute.SetBool("_FlipNormals", false);
            _gradientCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _gradientCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            
            // Set principal curvature compute uniforms
            _principalCurvatureCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            _principalCurvatureCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            
            // Set streamline builder compute uniforms
            _streamlineBuilderCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            _streamlineBuilderCompute.SetVector("_VoxelStartPosition", _sdf.StartPosition);
            _streamlineBuilderCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _streamlineBuilderCompute.SetBool("_ScaleLengthByCurvature", true);
            _streamlineBuilderCompute.SetBool("_ScaleWidthByCurvature", true);
            _streamlineBuilderCompute.SetBool("_Taper", true);
            
            // Set LIC compute uniforms
            _licCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            _licCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _licCompute.SetVector("_VoxelStartPosition", _sdf.StartPosition);
            
            // Set marching cubes uniforms
            _marchingCubesCompute.SetInts("_Dimensions", new int[3] { _sdf.Dimensions.x, _sdf.Dimensions.y, _sdf.Dimensions.z });
            _marchingCubesCompute.SetVector("_VoxelStartPosition", _sdf.StartPosition);
            _marchingCubesCompute.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _marchingCubesCompute.SetFloat("_IsoValue", 0);

            // Run all stages of the render pipeline
            RunGradientCompute();
            RunSurfacePointSampler();
            RunPoissonDiskSampler();
            RunCurvatureCompute();
            // RunMinMaxCurvatureCompute();
            // RunStreamlineBuilder();
            RunLIC();
            RunMarchingCubes();
        }
        
        /// <summary>
        /// Render the streamlines every frame in LateUpdate.
        /// </summary>
        private void LateUpdate()
        {
            if (!_initialized) return;
            
            // Update the model matrix
            _curvatureStreamlinesMaterial.SetMatrix("_ObjectToWorld", transform.localToWorldMatrix);
            _curvatureStreamlinesMaterial.SetMatrix("_WorldToObject", transform.worldToLocalMatrix);
            // Calculate the bounding box
            Bounds bounds = GeometryUtility.CalculateBounds(new []{_sdf.Bounds.min, _sdf.Bounds.max}, transform.localToWorldMatrix);
            // Draw the curvature streamlines indirectly
            Graphics.DrawProceduralIndirect(_curvatureStreamlinesMaterial, bounds, MeshTopology.Triangles, _drawArgsBuffer, 0, null, null, _shadowCastingMode, true, gameObject.layer);
        }

        /// <summary>
        /// Run the compute shader that performs sampling on the surface of the SDF.
        /// </summary>
        private void RunSurfacePointSampler()
        {
            if (!_initialized) return;
            
            // Sample points along the surface of the SDF
            _pointCounterBuffer.SetCounterValue(0);
            _groupCounterBuffer.SetCounterValue(0);
            _surfaceSamplerCompute.Dispatch(_sampleSurfaceKernel, _sampleSurfaceThreadGroups.x, _sampleSurfaceThreadGroups.y, _sampleSurfaceThreadGroups.z);
            
            // Record how many surface points were sampled            
            ComputeBuffer.CopyCount(_pointCounterBuffer, _surfacePointsCountBuffer, 0);
            // Record how many groups we should dispatch indirectly for all these points
            ComputeBuffer.CopyCount(_groupCounterBuffer, _poissonDispatchArgsBuffer, 0);
        }
        

        /// <summary>
        /// Run the compute shader that performs poisson disk sampling.
        /// </summary>
        private void RunPoissonDiskSampler()
        {
            if (!_initialized) return;

            // Pass poisson disk radius radius and max points per cell to compute
            _poissonCompute.SetInt("_MaxPointsPerCell", (int) (MAX_POISSON_POINTS_PER_VOXEL * Mathf.Pow(_poissonRadius / _sdf.VoxelSpacing.x, 2)));
            _poissonCompute.SetFloat("_RSqr", _poissonRadius * _poissonRadius);
            
            // Update poisson cell dimensions given poisson radius
            Vector3 cellDimensionsFloat = Vector3.Scale(_sdf.VoxelSpacing, _sdf.Dimensions) / _poissonRadius;
            Vector3Int cellDimensions = cellDimensionsFloat.CeilToInt();
            _poissonCompute.SetInts("_CellDimensions", new int[3] { cellDimensions.x, cellDimensions.y, cellDimensions.z });
            
            // Get thread groups for current cell dimensions / poisson radius
            Vector3Int poissonInitThreadGroups = ComputeUtilities.GetThreadGroups(_poissonInitThreadGroupsX, _poissonInitThreadGroupsY, _poissonInitThreadGroupsZ, cellDimensions);
            Vector3Int poissonRemoveThreadGroups = ComputeUtilities.GetThreadGroups(_poissonRemoveThreadGroupsX, _poissonRemoveThreadGroupsY, _poissonRemoveThreadGroupsZ, cellDimensions);
            Vector3Int poissonCollapseThreadGroups = ComputeUtilities.GetThreadGroups(_poissonCollapseThreadGroupsX, _poissonCollapseThreadGroupsY, _poissonCollapseThreadGroupsZ, cellDimensions);
      
            // Dispatch poisson disk sampler compute
            _pointCounterBuffer.SetCounterValue(0);
            _groupCounterBuffer.SetCounterValue(0);
            _poissonCompute.Dispatch(_poissonInitKernel, poissonInitThreadGroups.x, poissonInitThreadGroups.y, poissonInitThreadGroups.z);
            _poissonCompute.DispatchIndirect(_poissonSelectKernel, _poissonDispatchArgsBuffer);
            for (int i = 0; i < 27; i++)
            {
                _poissonCompute.SetInt("_Phase", i);
                _poissonCompute.Dispatch(_poissonRemoveKernel, poissonRemoveThreadGroups.x, poissonRemoveThreadGroups.y, poissonRemoveThreadGroups.z);
            }
            _poissonCompute.Dispatch(_poissonCollapseKernel, poissonCollapseThreadGroups.x, poissonCollapseThreadGroups.y, poissonCollapseThreadGroups.z);
            
            // Record how many poisson points were sampled            
            ComputeBuffer.CopyCount(_pointCounterBuffer, _poissonCountBuffer, 0);
            // Record how many groups we should dispatch indirectly for all these points
            ComputeBuffer.CopyCount(_groupCounterBuffer, _streamlineDispatchArgsBuffer, 0);
        }
        
        /// <summary>
        /// Run the compute shader that calculates the gradient at every voxel in the SDF.
        /// </summary>
        private void RunGradientCompute()
        {
            if (!_initialized) return;
            
            _gradientCompute.Dispatch(_gradientKernel, _gradientThreadGroups.x, _gradientThreadGroups.y, _gradientThreadGroups.z);
        }

        /// <summary>
        /// Run the compute shader that calculates the principal curvature at every voxel in the SDF.
        /// </summary>
        private void RunCurvatureCompute()
        {
            if (!_initialized) return;
            
            _principalCurvatureCompute.Dispatch(_curvatureKernel, _curvatureThreadGroups.x, _curvatureThreadGroups.y, _curvatureThreadGroups.z);
        }

        /// <summary>
        /// Run the compute shader that calculates the min/max curvature for the selected poisson points.
        /// </summary>
        private void RunMinMaxCurvatureCompute()
        {
            if (!_initialized) return;
            
            // Reset curvature min max
            _curvatureMinMaxBuffer.SetData(_minMaxBufferReset);

            // Dispatch
            _streamlineBuilderCompute.DispatchIndirect(_minMaxCurvatureKernel, _streamlineDispatchArgsBuffer);
        }
        
        /// <summary>
        /// Run the compute shader that builds streamlines. 
        /// </summary>
        private void RunStreamlineBuilder()
        {
            if (!_initialized) return;
            
            // Reset triangle and draw args
            _streamlineTriangleBuffer.SetCounterValue(0);
            _drawArgsBuffer.SetData(_drawArgsBufferReset);
            
            if (_usePoisson)
            {
                _streamlineBuilderCompute.DispatchIndirect(_streamlineKernel, _streamlineDispatchArgsBuffer);
            }
            // For demonstration purposes only, hack to inject non-poisson points as seeds
            // for streamline building
            else
            {
                // Get original surface point sample set
                int[] surfacePointCountBuffer = new int[1];
                _surfacePointsCountBuffer.GetData(surfacePointCountBuffer);
                int surfacePointCount = surfacePointCountBuffer[0];
                Vector3[] surfacePoints = new Vector3[surfacePointCount];
                _surfacePointsBuffer.GetData(surfacePoints);
                
                // Use reservoir sampling to randomly sample from original point set
                int pointCount = 20000;

                int i;
                Vector3[] reservoir = new Vector3[pointCount];
                for (i = 0; i < pointCount; i++)
                    reservoir[i] = surfacePoints[i];

                for (; i < surfacePoints.Length; i++)
                {
                    int j = Random.Range(0, i + 1);
                    if (j < pointCount)
                        reservoir[j] = surfacePoints[i];
                }
                
                // Pass newly sampled random points to streamline builder
                _poissonBuffer.SetData(reservoir);
                _pointCounterBuffer.SetCounterValue((uint)pointCount);
                
                // Dispatch
                _streamlineBuilderCompute.Dispatch(_streamlineKernel, Mathf.CeilToInt((float)pointCount / _streamlineThreadGroupsX), 1, 1);
            }
        }

        /// <summary>
        /// Run the compute shader that performs line integral convolution (LIC). 
        /// </summary>
        private void RunLIC()
        {
            if (!_initialized) return;
            
            _licCompute.Dispatch(_noiseKernel, _noiseThreadGroups.x, _noiseThreadGroups.y, _noiseThreadGroups.z);
            
            _licCompute.DispatchIndirect(_particlesKernel, _streamlineDispatchArgsBuffer);
            
            _licCompute.Dispatch(_licKernel, _licThreadGroups.x, _licThreadGroups.y, _licThreadGroups.z);
        }

        /// <summary>
        /// Run the compute shader that performs marching cubes
        /// </summary>
        private void RunMarchingCubes()
        {
            if (!_initialized) return;
            
            _streamlineTriangleBuffer.SetCounterValue(0);
            _drawArgsBuffer.SetData(_drawArgsBufferReset);
            
            _marchingCubesCompute.Dispatch(_marchKernel, _mcThreadGroups.x, _mcThreadGroups.y, _mcThreadGroups.z);
        }

        // Taper
        public void TaperStreamlines(bool taper)
        {
            _streamlineBuilderCompute.SetBool("_Taper", taper);
            RunStreamlineBuilder();
        }
        // Width
        public void ScaleStreamlineWidthByCurvature(bool scaleByCurvature)
        {
            _streamlineBuilderCompute.SetBool("_ScaleWidthByCurvature", scaleByCurvature);
            RunStreamlineBuilder();
        }
        public void UpdateStreamlineWidth(float width)
        {
            _streamlineBuilderCompute.SetFloat("_Width", width/5); // For demo purposes only
            RunStreamlineBuilder();
        }
        public void UpdateStreamlineMinWidth(float minWidth)
        {
            _streamlineBuilderCompute.SetFloat("_MinWidth", minWidth);
            RunStreamlineBuilder();
        }
        public void UpdateStreamlineMaxWidth(float maxWidth)
        {
            _streamlineBuilderCompute.SetFloat("_MaxWidth", maxWidth);
            RunStreamlineBuilder();
        }
        // Length
        public void ScaleStreamlineLengthByCurvature(bool scaleByCurvature)
        {
            _streamlineBuilderCompute.SetBool("_ScaleLengthByCurvature", scaleByCurvature);
            RunStreamlineBuilder();
        }
        public void UpdateStreamlineLength(float length)
        {
            _streamlineBuilderCompute.SetFloat("_Length", length/5); // For demo purposes only
            RunStreamlineBuilder();
        }
        public void UpdateStreamlineMinLength(float minlength)
        {
            _streamlineBuilderCompute.SetFloat("_MinLength", minlength);
            RunStreamlineBuilder();
        }
        public void UpdateStreamlineMaxLength(float maxlength)
        {
            _streamlineBuilderCompute.SetFloat("_MaxLength", maxlength);
            RunStreamlineBuilder();
        }
        // Spacing
        public void UpdatePoissonRadius(float radius)
        {
            _poissonRadius = Mathf.Max(_sdf.VoxelSpacing.x, radius);
            RunPoissonDiskSampler();
            // RunMinMaxCurvatureCompute();
            // RunStreamlineBuilder();
            RunLIC();
        }
        // Curvature scale
        public void UpdateCurvatureScale(float curvatureScale)
        {
            _principalCurvatureCompute.SetFloat("_CurvatureScale", curvatureScale);
            RunCurvatureCompute();
            // RunMinMaxCurvatureCompute();
            // RunStreamlineBuilder();
            RunLIC();
        }
        // Scale by curvature
        public void ScaleByCurvature(bool scaleByCurvature)
        {
            _streamlineBuilderCompute.SetBool("_ScaleLengthByCurvature", scaleByCurvature);
            _streamlineBuilderCompute.SetBool("_ScaleWidthByCurvature", scaleByCurvature);
            RunStreamlineBuilder();
        }
        // Use poisson
        public void UsePoisson(bool usePoisson)
        {
            _usePoisson = usePoisson;
            if (_usePoisson)
            {
                RunPoissonDiskSampler();
                RunMinMaxCurvatureCompute();
                RunStreamlineBuilder();
            }
            else
            {
                RunStreamlineBuilder();
            }
        }

        /// <summary>
        /// Initializes all compute buffers needed to generate and render streamlines.
        /// </summary>
        private void CreateBuffers()
        {
            ReleaseBuffers();

            int numVoxels = _sdf.Dimensions.x * _sdf.Dimensions.y * _sdf.Dimensions.z;
            int maxTriangleCount = (_sdf.Dimensions.x-1) * (_sdf.Dimensions.y-1) * (_sdf.Dimensions.z-1);

            _streamlineTriangleBuffer = new ComputeBuffer(maxTriangleCount, DRAW_TRIANGLE_STRIDE, ComputeBufferType.Append);
            _voxelBuffer = new ComputeBuffer(numVoxels, VOXEL_STRIDE, ComputeBufferType.Default);
            _drawArgsBuffer = new ComputeBuffer(1, DRAW_INDIRECT_ARGS_STRIDE, ComputeBufferType.IndirectArguments);
            _poissonDispatchArgsBuffer = new ComputeBuffer(1, DISPATCH_INDIRECT_ARGS_STRIDE, ComputeBufferType.IndirectArguments);
            _streamlineDispatchArgsBuffer = new ComputeBuffer(1, DISPATCH_INDIRECT_ARGS_STRIDE, ComputeBufferType.IndirectArguments);

            _gradientBuffer = new ComputeBuffer(numVoxels, VOXEL_STRIDE * 3, ComputeBufferType.Default);
            _surfacePointsBuffer = new ComputeBuffer(numVoxels, POINT_STRIDE, ComputeBufferType.Default);
            _surfacePointsCountBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Raw);
            _pointCounterBuffer = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Counter);
            _groupCounterBuffer = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Counter);
            _cellsBuffer = new ComputeBuffer(numVoxels, sizeof(int), ComputeBufferType.Default);
            _pointsByCellBuffer = new ComputeBuffer(numVoxels * MAX_POISSON_POINTS_PER_VOXEL, sizeof(int), ComputeBufferType.Default);
            _poissonBuffer = new ComputeBuffer(numVoxels, POINT_STRIDE, ComputeBufferType.Default);
            _poissonCountBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Raw);
            _principalCurvatureBuffer = new ComputeBuffer(numVoxels, VOXEL_STRIDE * 4, ComputeBufferType.Default);
            _curvatureMinMaxBuffer = new ComputeBuffer(1, MIN_MAX_STRIDE, ComputeBufferType.Default);
            _noiseBuffer = new ComputeBuffer(numVoxels, VOXEL_STRIDE, ComputeBufferType.Default);
            _licBuffer = new ComputeBuffer(numVoxels, VOXEL_STRIDE, ComputeBufferType.Default);

            _triangleTableBuffer = new ComputeBuffer(256, sizeof(ulong));
            _triangleTableBuffer.SetData(TRIANGLE_TABLE);
        }

        /// <summary>
        /// Releases all buffers to avoid memory leaks.
        /// </summary>
        void ReleaseBuffers()
        {
            _streamlineTriangleBuffer?.Release();
            _voxelBuffer?.Release();
            _drawArgsBuffer?.Release();
            _poissonDispatchArgsBuffer?.Release();
            _streamlineDispatchArgsBuffer?.Release();
            _gradientBuffer?.Release();
            _surfacePointsBuffer?.Release();
            _surfacePointsCountBuffer?.Release();
            _pointCounterBuffer?.Release();
            _groupCounterBuffer?.Release();
            _cellsBuffer?.Release();
            _pointsByCellBuffer?.Release();
            _principalCurvatureBuffer?.Release();
            _curvatureMinMaxBuffer?.Release();
            _poissonBuffer?.Release();
            _poissonCountBuffer?.Release();
            _noiseBuffer?.Release();
            _licBuffer?.Release();
            _triangleTableBuffer?.Release();

            _streamlineTriangleBuffer = null;
            _streamlineDispatchArgsBuffer = null;
            _voxelBuffer = null;
            _drawArgsBuffer = null;
            _poissonDispatchArgsBuffer = null;
            _gradientBuffer = null;
            _surfacePointsBuffer = null;
            _surfacePointsCountBuffer = null;
            _pointCounterBuffer = null;
            _groupCounterBuffer = null;
            _cellsBuffer = null;
            _pointsByCellBuffer = null;
            _principalCurvatureBuffer = null;
            _curvatureMinMaxBuffer = null;
            _poissonBuffer = null;
            _poissonCountBuffer = null;
            _noiseBuffer = null;
            _licBuffer = null;
            _triangleTableBuffer = null;
        }
        
        /// <summary>
        /// Attaches buffers to their respective compute shaders/materials.
        /// </summary>
        private void SetBuffers()
        {
            // Material buffers
            _curvatureStreamlinesMaterial.SetBuffer("_DrawTriangles", _streamlineTriangleBuffer);
            _curvatureStreamlinesMaterial.SetBuffer("_LIC", _licBuffer);
            _curvatureStreamlinesMaterial.SetVector("_Dimensions", (Vector3) _sdf.Dimensions);
            _curvatureStreamlinesMaterial.SetVector("_VoxelSpacing", _sdf.VoxelSpacing);
            _curvatureStreamlinesMaterial.SetVector("_VoxelStartPosition", _sdf.StartPosition);
            
            // Gradients buffers
            _gradientCompute.SetBuffer(_gradientKernel, "_Voxels", _voxelBuffer);
            _gradientCompute.SetBuffer(_gradientKernel, "_Gradients", _gradientBuffer);
            
            // Surface sampling buffers
            _surfaceSamplerCompute.SetBuffer(_sampleSurfaceKernel, "_Voxels", _voxelBuffer);
            _surfaceSamplerCompute.SetBuffer(_sampleSurfaceKernel, "_SurfacePoints", _surfacePointsBuffer);
            _surfaceSamplerCompute.SetBuffer(_sampleSurfaceKernel, "_PointCounterBuffer", _pointCounterBuffer);
            _surfaceSamplerCompute.SetBuffer(_sampleSurfaceKernel, "_ThreadGroupCounterBuffer", _groupCounterBuffer);
            
            // Poisson buffers
            _poissonCompute.SetBuffer(_poissonInitKernel, "_Cells", _cellsBuffer);
            _poissonCompute.SetBuffer(_poissonInitKernel, "_PointsByCell", _pointsByCellBuffer);
            
            _poissonCompute.SetBuffer(_poissonSelectKernel, "_SurfacePointsCount", _surfacePointsCountBuffer);
            _poissonCompute.SetBuffer(_poissonSelectKernel, "_SurfacePoints", _surfacePointsBuffer);
            _poissonCompute.SetBuffer(_poissonSelectKernel, "_PointsByCell", _pointsByCellBuffer);
            
            _poissonCompute.SetBuffer(_poissonRemoveKernel, "_SurfacePoints", _surfacePointsBuffer);
            _poissonCompute.SetBuffer(_poissonRemoveKernel, "_Cells", _cellsBuffer);
            _poissonCompute.SetBuffer(_poissonRemoveKernel, "_PointsByCell", _pointsByCellBuffer);
            
            _poissonCompute.SetBuffer(_poissonCollapseKernel, "_SurfacePoints", _surfacePointsBuffer);
            _poissonCompute.SetBuffer(_poissonCollapseKernel, "_Cells", _cellsBuffer);
            _poissonCompute.SetBuffer(_poissonCollapseKernel, "_PoissonPoints", _poissonBuffer);
            _poissonCompute.SetBuffer(_poissonCollapseKernel, "_PointCounterBuffer", _pointCounterBuffer);
            _poissonCompute.SetBuffer(_poissonCollapseKernel, "_ThreadGroupCounterBuffer", _groupCounterBuffer);
            
            // Principal curvature buffers
            _principalCurvatureCompute.SetBuffer(_curvatureKernel, "_Voxels", _voxelBuffer);
            _principalCurvatureCompute.SetBuffer(_curvatureKernel, "_Gradients", _gradientBuffer);
            _principalCurvatureCompute.SetBuffer(_curvatureKernel, "_PrincipalCurvatures", _principalCurvatureBuffer);
            
            // Streamline builder buffers
            _streamlineBuilderCompute.SetBuffer(_minMaxCurvatureKernel, "_PrincipalCurvatures", _principalCurvatureBuffer);
            _streamlineBuilderCompute.SetBuffer(_minMaxCurvatureKernel, "_PoissonPoints", _poissonBuffer);
            _streamlineBuilderCompute.SetBuffer(_minMaxCurvatureKernel, "_PoissonPointsCount", _poissonCountBuffer);
            _streamlineBuilderCompute.SetBuffer(_minMaxCurvatureKernel, "_CurvatureMinMax", _curvatureMinMaxBuffer);
            
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_Voxels", _voxelBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_Gradients", _gradientBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_PrincipalCurvatures", _principalCurvatureBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_IndirectArgs", _drawArgsBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_DrawTriangles", _streamlineTriangleBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_PoissonPoints", _poissonBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_PoissonPointsCount", _poissonCountBuffer);
            _streamlineBuilderCompute.SetBuffer(_streamlineKernel, "_CurvatureMinMax", _curvatureMinMaxBuffer);
            
            // LIC buffers
            _licCompute.SetBuffer(_particlesKernel, "_WhiteNoise", _noiseBuffer);
            _licCompute.SetBuffer(_particlesKernel, "_PoissonPointsCount", _poissonCountBuffer);
            _licCompute.SetBuffer(_particlesKernel, "_PoissonPoints", _poissonBuffer);
            
            _licCompute.SetBuffer(_noiseKernel, "_Voxels", _voxelBuffer);
            _licCompute.SetBuffer(_noiseKernel, "_WhiteNoise", _noiseBuffer);
            
            _licCompute.SetBuffer(_licKernel, "_Voxels", _voxelBuffer);
            _licCompute.SetBuffer(_licKernel, "_Gradients", _gradientBuffer);
            _licCompute.SetBuffer(_licKernel, "_WhiteNoise", _noiseBuffer);
            _licCompute.SetBuffer(_licKernel, "_LIC", _licBuffer);
            _licCompute.SetBuffer(_licKernel, "_PrincipalCurvatures", _principalCurvatureBuffer);
            
            // Marching cubes buffers
            _marchingCubesCompute.SetBuffer(_marchKernel, "_Voxels", _voxelBuffer);
            _marchingCubesCompute.SetBuffer(_marchKernel, "_Gradients", _gradientBuffer);
            _marchingCubesCompute.SetBuffer(_marchKernel, "_IndirectArgs", _drawArgsBuffer);
            _marchingCubesCompute.SetBuffer(_marchKernel, "_DrawTriangles", _streamlineTriangleBuffer);
            _marchingCubesCompute.SetBuffer(_marchKernel, "_TriangleTable", _triangleTableBuffer);
        }
        
        /// <summary>
        /// Re-attach buffers on application focus since buffer connections
        /// are lost when the application loses focus.
        /// </summary>
        /// <param name="hasFocus"></param>
        private void OnApplicationFocus(bool hasFocus)
        {
            if (hasFocus && _initialized)
            {
                SetBuffers();
            }
        }

        /// <summary>
        /// Release all buffers when destroyed.
        /// </summary>
        private void OnDestroy()
        {
            ReleaseBuffers();
        }
        
        private ulong [] TRIANGLE_TABLE =
        {
            0xffffffffffffffffUL,
            0xfffffffffffff380UL,
            0xfffffffffffff910UL,
            0xffffffffff189381UL,
            0xfffffffffffffa21UL,
            0xffffffffffa21380UL,
            0xffffffffff920a29UL,
            0xfffffff89a8a2382UL,
            0xfffffffffffff2b3UL,
            0xffffffffff0b82b0UL,
            0xffffffffffb32091UL,
            0xfffffffb89b912b1UL,
            0xffffffffff3ab1a3UL,
            0xfffffffab8a801a0UL,
            0xfffffff9ab9b3093UL,
            0xffffffffffb8aa89UL,
            0xfffffffffffff874UL,
            0xffffffffff437034UL,
            0xffffffffff748910UL,
            0xfffffff137174914UL,
            0xffffffffff748a21UL,
            0xfffffffa21403743UL,
            0xfffffff748209a29UL,
            0xffff4973727929a2UL,
            0xffffffffff2b3748UL,
            0xfffffff40242b74bUL,
            0xfffffffb32748109UL,
            0xffff1292b9b49b74UL,
            0xfffffff487ab31a3UL,
            0xffff4b7401b41ab1UL,
            0xffff30bab9b09874UL,
            0xfffffffab99b4b74UL,
            0xfffffffffffff459UL,
            0xffffffffff380459UL,
            0xffffffffff051450UL,
            0xfffffff513538458UL,
            0xffffffffff459a21UL,
            0xfffffff594a21803UL,
            0xfffffff204245a25UL,
            0xffff8434535235a2UL,
            0xffffffffffb32459UL,
            0xfffffff594b802b0UL,
            0xfffffffb32510450UL,
            0xffff584b82852512UL,
            0xfffffff45931ab3aUL,
            0xffffab81a8180594UL,
            0xffff30bab5b05045UL,
            0xfffffffb8aa85845UL,
            0xffffffffff975879UL,
            0xfffffff375359039UL,
            0xfffffff751710870UL,
            0xffffffffff753351UL,
            0xfffffff21a759879UL,
            0xffff37503505921aUL,
            0xffff25a758528208UL,
            0xfffffff7533525a2UL,
            0xfffffff2b3987597UL,
            0xffffb72029279759UL,
            0xffff751871810b32UL,
            0xfffffff51771b12bUL,
            0xffffb3a31a758859UL,
            0xf0aba010b7905075UL,
            0xf07570805a30b0abUL,
            0xffffffffff5b75abUL,
            0xfffffffffffff56aUL,
            0xffffffffff6a5380UL,
            0xffffffffff6a5109UL,
            0xfffffff6a5891381UL,
            0xffffffffff162561UL,
            0xfffffff803621561UL,
            0xfffffff620609569UL,
            0xffff823625285895UL,
            0xffffffffff56ab32UL,
            0xfffffff56a02b80bUL,
            0xfffffff6a5b32910UL,
            0xffffb892b92916a5UL,
            0xfffffff315356b36UL,
            0xffff6b51505b0b80UL,
            0xffff9505606306b3UL,
            0xfffffff89bb96956UL,
            0xffffffffff8746a5UL,
            0xfffffffa56374034UL,
            0xfffffff7486a5091UL,
            0xffff49737179156aUL,
            0xfffffff874156216UL,
            0xffff743403625521UL,
            0xffff620560509748UL,
            0xf962695923497937UL,
            0xfffffff56a4872b3UL,
            0xffffb720242746a5UL,
            0xffff6a5b32874910UL,
            0xf6a54b7b492b9129UL,
            0xffff6b51535b3748UL,
            0xfb404b7b016b5b15UL,
            0xf74836b630560950UL,
            0xffff9b7974b96956UL,
            0xffffffffffa4694aUL,
            0xfffffff380a946a4UL,
            0xfffffff04606a10aUL,
            0xffffa16468618138UL,
            0xfffffff462421941UL,
            0xffff462942921803UL,
            0xffffffffff624420UL,
            0xfffffff624428238UL,
            0xfffffff32b46a94aUL,
            0xffff6a4a94b82280UL,
            0xffffa164606102b3UL,
            0xf1b8b12184a16146UL,
            0xffff36b319639469UL,
            0xf14641916b0181b8UL,
            0xfffffff4600636b3UL,
            0xffffffffff86b846UL,
            0xfffffffa98a876a7UL,
            0xffffa76a907a0370UL,
            0xffff0818717a176aUL,
            0xfffffff37117a76aUL,
            0xffff768981861621UL,
            0xf937390976192962UL,
            0xfffffff206607087UL,
            0xffffffffff276237UL,
            0xffff76898a86ab32UL,
            0xf7a9a76790b72702UL,
            0xfb32a767a1871081UL,
            0xffff17616a71b12bUL,
            0xf63136b619768698UL,
            0xffffffffff76b190UL,
            0xffff06b0b3607087UL,
            0xfffffffffffff6b7UL,
            0xfffffffffffffb67UL,
            0xffffffffff67b803UL,
            0xffffffffff67b910UL,
            0xfffffff67b138918UL,
            0xffffffffff7b621aUL,
            0xfffffff7b6803a21UL,
            0xfffffff7b69a2092UL,
            0xffff89a38a3a27b6UL,
            0xffffffffff726327UL,
            0xfffffff026067807UL,
            0xfffffff910732672UL,
            0xffff678891681261UL,
            0xfffffff73171a67aUL,
            0xffff801781a7167aUL,
            0xffff7a69a0a70730UL,
            0xfffffff9a88a7a67UL,
            0xffffffffff68b486UL,
            0xfffffff640603b63UL,
            0xfffffff109648b68UL,
            0xffff63b139369649UL,
            0xfffffff1a28b6486UL,
            0xffff640b60b03a21UL,
            0xffff9a2920b648b4UL,
            0xf36463b34923a39aUL,
            0xfffffff264248328UL,
            0xffffffffff264240UL,
            0xffff834642432091UL,
            0xfffffff642241491UL,
            0xffff1a6648168318UL,
            0xfffffff40660a01aUL,
            0xf39a9303a6834364UL,
            0xffffffffff4a649aUL,
            0xffffffffffb67594UL,
            0xfffffff67b594380UL,
            0xfffffffb67045105UL,
            0xffff51345343867bUL,
            0xfffffffb6721a459UL,
            0xffff594380a217b6UL,
            0xffff204a24a45b67UL,
            0xf67b25a523453843UL,
            0xfffffff945267327UL,
            0xffff786260680459UL,
            0xffff045051673263UL,
            0xf851584812786826UL,
            0xffff73167161a459UL,
            0xf459078701671a61UL,
            0xfa737a6a305a4a04UL,
            0xffffa84a458a7a67UL,
            0xfffffff98b9b6596UL,
            0xffff590650360b63UL,
            0xffffb65510b508b0UL,
            0xfffffff1355363b6UL,
            0xffff65b8b9b59a21UL,
            0xfa21965690b603b0UL,
            0xf52025a50865b58bUL,
            0xffff35a3a25363b6UL,
            0xffff283265825985UL,
            0xfffffff260069659UL,
            0xf826283865081851UL,
            0xffffffffff612651UL,
            0xf698965683a61631UL,
            0xffff06505960a01aUL,
            0xffffffffffa65830UL,
            0xfffffffffffff65aUL,
            0xffffffffffb57a5bUL,
            0xfffffff03857ba5bUL,
            0xfffffff091ba57b5UL,
            0xffff1381897ba57aUL,
            0xfffffff15717b21bUL,
            0xffffb27571721380UL,
            0xffff7b2209729579UL,
            0xf289823295b27257UL,
            0xfffffff573532a52UL,
            0xffff52a578258028UL,
            0xffff2a37353a5109UL,
            0xf25752a278129289UL,
            0xffffffffff573531UL,
            0xfffffff571170780UL,
            0xfffffff735539309UL,
            0xffffffffff795789UL,
            0xfffffff8ba8a5485UL,
            0xffff03bba50b5405UL,
            0xffff54aba8a48910UL,
            0xf41314943b54a4baUL,
            0xffff8548b2582152UL,
            0xfb151b2b543b0b40UL,
            0xf58b8545b2950520UL,
            0xffffffffff3b2549UL,
            0xffff483543253a52UL,
            0xfffffff0244252a5UL,
            0xf910854583a532a3UL,
            0xffff2492914252a5UL,
            0xfffffff153358548UL,
            0xffffffffff501540UL,
            0xffff530509358548UL,
            0xfffffffffffff549UL,
            0xfffffffba9b947b4UL,
            0xffffba97b9794380UL,
            0xffffb470414b1ba1UL,
            0xf4bab474a1843413UL,
            0xffff219b294b97b4UL,
            0xf3801b2b197b9479UL,
            0xfffffff04224b47bUL,
            0xffff42343824b47bUL,
            0xffff947732972a92UL,
            0xf70207872a4797a9UL,
            0xfa040a1a472a3a73UL,
            0xffffffffff4782a1UL,
            0xfffffff317714194UL,
            0xffff178180714194UL,
            0xffffffffff347304UL,
            0xfffffffffffff784UL,
            0xffffffffff8ba8a9UL,
            0xfffffffa9bb93903UL,
            0xfffffffba88a0a10UL,
            0xffffffffffa3ba13UL,
            0xfffffff8b99b1b21UL,
            0xffff9b2921b93903UL,
            0xffffffffffb08b20UL,
            0xfffffffffffffb23UL,
            0xfffffff98aa82832UL,
            0xffffffffff2902a9UL,
            0xffff8a1810a82832UL,
            0xfffffffffffff2a1UL,
            0xffffffffff819831UL,
            0xfffffffffffff190UL,
            0xfffffffffffff830UL,
            0xffffffffffffffffUL
        };
    }
}
