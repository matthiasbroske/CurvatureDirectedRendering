using System.IO;
using Matthias.Utilities;
using UnityEngine;
using UnityEngine.Events;

namespace Curvature
{
    /// <summary>
    /// Simple helper class for loading binary SDF data from disk.
    /// </summary>
    public class SDFLoader : MonoBehaviour
    {
        [Header("File Path")]
        [SerializeField] private string _filePath = "SDFs";
        [SerializeField] private string _fileName;
        [Header("SDF Info")]
        [SerializeField] private Vector3Int _dimensions;
        [SerializeField] private Vector3 _size;
        [Header("Callback")]
        [SerializeField] private UnityEvent<SDF> _onSDFLoaded;
        
        void Start()
        {
            // Calculate the voxel spacing
            Vector3 voxelSpacing = new Vector3(_size.x / _dimensions.x, _size.y / _dimensions.y, _size.z / _dimensions.z);
            // Create the sdf
            SDF sdf = new SDF(Vector3.zero, voxelSpacing, Vector3Int.one * _dimensions, new float[_dimensions.x*_dimensions.y*_dimensions.z]);
            // Load the binary voxel data
            byte[] byteData = File.ReadAllBytes(Path.Combine(_filePath, _fileName + ".sdf"));
            System.Buffer.BlockCopy(byteData, 0, sdf.Voxels, 0, byteData.Length);
            // Invoke loaded callback
            _onSDFLoaded?.Invoke(sdf);
        }
    }
}
