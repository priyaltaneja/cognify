**Cognify - Medical-grade brain volumetric analysis for early detection of Alzheimer's disease and neurodegeneration.**

Cognify is an AI-powered web application that analyzes brain MRI scans to detect early signs of neurodegeneration, with a focus on Alzheimer's disease and Mild Cognitive Impairment (MCI).

## Features

### Medical-Grade Biomarkers
- **Hippocampal Occupancy Score (HOC)** - Early biomarker for medial temporal atrophy
- **Brain Parenchymal Fraction (BPF)** - Overall brain tissue integrity measure
- **Intracranial Volume (ICV)** - Age/sex-adjusted head size normalization

### Standardized Clinical Scales
- **MTA Score** (Scheltens Scale) - Medial Temporal Atrophy grading (0-4)
- **GCA Score** (Pasquier Scale) - Global Cortical Atrophy grading (0-3)
- **Koedam Score** - Posterior Atrophy assessment
- **Evans Index** - Ventricular enlargement indicator

### Clinical Pattern Recognition
- Alzheimer's Disease pattern detection
- Frontotemporal Dementia indicators
- Early MCI detection via HOC
- Vascular dementia markers
- Normal aging differentiation

### Regional Analysis
- 17 brain regions analyzed with age/sex-matched normative data
- Z-scores and percentiles for each region
- ICV-normalized volumes using residual correction method

## Technology

- **AI Segmentation**: TensorFlow.js neural network for brain parcellation
- **Visualization**: NiiVue for interactive MRI viewing
- **Normative Data**: Based on UK Biobank (n=19,793), ADNI, and FreeSurfer norms

## Usage

1. Load a T1-weighted brain MRI (NIfTI format: .nii or .nii.gz)
2. Enter patient age and sex
3. Click "Analyze" for AI-powered segmentation and analysis
4. Review medical-grade biomarkers and clinical recommendations

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Credits & Acknowledgments

### Built on BrainChop

This project is built upon [**BrainChop**](https://github.com/neuroneural/brainchop) by the [neuroneural](https://github.com/neuroneural) team.

BrainChop provides the core AI-powered brain segmentation capabilities using TensorFlow.js. We are grateful for their open-source contribution to neuroimaging.

**Original BrainChop Paper:**
> Masoud et al. "BrainChop: In-browser MRI Volumetry using Deep Learning" - [neuroneural/brainchop](https://github.com/neuroneural/brainchop)

### Normative Data Sources

- **UK Biobank** - Nobis et al. (2019) Hippocampal nomograms (n=19,793)
- **ADNI** - Alzheimer's Disease Neuroimaging Initiative
- **FreeSurfer** - Potvin et al. (2016) Subcortical normative data (n=2,713)
- **BPF Norms** - Vågberg et al. (2017) Systematic review (n=9,269)
- **NeuroQuant/Cortechs.ai** - FDA-cleared reference standards

### Clinical Scales

- **MTA Score**: Scheltens et al. (1992)
- **GCA Score**: Pasquier et al. (1996)
- **Koedam Score**: Koedam et al. (2011)
- **Evans Index**: Evans (1942)

## Disclaimer

**For Research and Educational Use Only**

This tool is intended for research and educational purposes. It is not a medical device and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Cognify - Advancing early detection of Alzheimer's disease through AI-powered neuroimaging analysis.*
