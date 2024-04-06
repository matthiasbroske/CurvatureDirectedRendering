using Matthias.Utilities;
using UnityEngine;
using UnityEngine.UI;

namespace Curvature
{
    /// <summary>
    /// <see cref="CurvatureRenderer"/> initialization helper.
    /// </summary>
    [RequireComponent(typeof(CurvatureRenderer))]
    public class CurvatureRendererInitializer : MonoBehaviour
    {
        [SerializeField] private Slider _curvatureSlider;
        [SerializeField] private Slider _lengthSlider;
        [SerializeField] private Slider _widthSlider;
        [SerializeField] private Slider _spacingSlider;

        /// <summary>
        /// Initializes the curvature renderer using the current values of the sliders.
        /// </summary>
        /// <param name="sdf">SDF to render</param>
        public void Initialize(SDF sdf)
        {
            CurvatureRenderer curvatureRenderer = GetComponent<CurvatureRenderer>();
            curvatureRenderer.Init(sdf, _curvatureSlider.value, _lengthSlider.value, _widthSlider.value, _spacingSlider.value);
            curvatureRenderer.Render();
        }
    }
}
