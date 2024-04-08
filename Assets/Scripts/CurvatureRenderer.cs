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
        // Thread groups
        private Vector3Int _gradientThreadGroups;
        private Vector3Int _sampleSurfaceThreadGroups;
        private uint _poissonInitThreadGroupsX, _poissonInitThreadGroupsY, _poissonInitThreadGroupsZ;
        private uint _poissonSelectThreadGroupsX;
        private uint _poissonRemoveThreadGroupsX, _poissonRemoveThreadGroupsY, _poissonRemoveThreadGroupsZ;
        private uint _poissonCollapseThreadGroupsX, _poissonCollapseThreadGroupsY, _poissonCollapseThreadGroupsZ;
        private Vector3Int _curvatureThreadGroups;
        private uint _streamlineThreadGroupsX;
        
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
            _streamlineBuilderCompute.SetFloat("_Length", length);
            _streamlineBuilderCompute.SetFloat("_MaxWidth", width);
            _streamlineBuilderCompute.SetFloat("_Width", width);
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
            
            // Get thread groups
            _gradientThreadGroups = ComputeUtilities.GetThreadGroups(_gradientCompute, _gradientKernel, _sdf.Dimensions);
            _sampleSurfaceThreadGroups = ComputeUtilities.GetThreadGroups(_surfaceSamplerCompute, _sampleSurfaceKernel, _sdf.Dimensions);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonInitKernel, out _poissonInitThreadGroupsX, out _poissonInitThreadGroupsY, out _poissonInitThreadGroupsZ);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonSelectKernel, out _poissonSelectThreadGroupsX, out _, out _);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonRemoveKernel, out _poissonRemoveThreadGroupsX, out _poissonRemoveThreadGroupsY, out _poissonRemoveThreadGroupsZ);
            _poissonCompute.GetKernelThreadGroupSizes(_poissonCollapseKernel, out _poissonCollapseThreadGroupsX, out _poissonCollapseThreadGroupsY, out _poissonCollapseThreadGroupsZ);
            _curvatureThreadGroups = ComputeUtilities.GetThreadGroups(_principalCurvatureCompute, _curvatureKernel, _sdf.Dimensions);
            _streamlineBuilderCompute.GetKernelThreadGroupSizes(_streamlineKernel, out _streamlineThreadGroupsX, out _, out _);
            
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
            _streamlineBuilderCompute.SetFloat("_Width", _sdf.VoxelSpacing.x);
            _streamlineBuilderCompute.SetBool("_ScaleLengthByCurvature", true);
            _streamlineBuilderCompute.SetBool("_ScaleWidthByCurvature", true);
            _streamlineBuilderCompute.SetBool("_Taper", true);

            // Run all stages of the render pipeline
            RunGradientCompute();
            RunSurfacePointSampler();
            RunPoissonDiskSampler();
            RunCurvatureCompute();
            RunMinMaxCurvatureCompute();
            RunStreamlineBuilder();
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
            
            // Dispatch
            _streamlineBuilderCompute.DispatchIndirect(_streamlineKernel, _streamlineDispatchArgsBuffer);
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
            _streamlineBuilderCompute.SetFloat("_Width", width);
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
            _streamlineBuilderCompute.SetFloat("_Length", length);
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
            RunMinMaxCurvatureCompute();
            RunStreamlineBuilder();
        }
        // Curvature scale
        public void UpdateCurvatureScale(float curvatureScale)
        {
            _principalCurvatureCompute.SetFloat("_CurvatureScale", curvatureScale);
            RunCurvatureCompute();
            RunMinMaxCurvatureCompute();
            RunStreamlineBuilder();
        }
        // Scale by curvature
        public void ScaleByCurvature(bool scaleByCurvature)
        {
            _streamlineBuilderCompute.SetBool("_ScaleLengthByCurvature", scaleByCurvature);
            _streamlineBuilderCompute.SetBool("_ScaleWidthByCurvature", scaleByCurvature);
            RunStreamlineBuilder();
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
        }
        
        /// <summary>
        /// Attaches buffers to their respective compute shaders/materials.
        /// </summary>
        private void SetBuffers()
        {
            // Material buffers
            _curvatureStreamlinesMaterial.SetBuffer("_DrawTriangles", _streamlineTriangleBuffer);
            
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
    }
}
