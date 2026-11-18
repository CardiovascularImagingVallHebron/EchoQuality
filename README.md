# EchoQuality: Automated echocardiography image quality assessment and its association with AI performance

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Paper%20coming%20soon-B31B1B?style=for-the-badge&logo=googlescholar&logoColor=white" alt="arXiv paper coming soon">
  <a href="https://github.com/CardiovascularImagingVallHebron/EchoQuality"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub code"></a>
  <a href="https://huggingface.co/CardiovascularImagingVHIR/EchoQuality"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face weights"></a>
</p>

EchoQuality is a deep-learning model for automated and reproducible assessment of echocardiographic image quality.

---

![EchoQuality overview](content/echoquality_overview.png)

## About

Developed from 7,475 expert-annotated videos, EchoQuality was evaluated across five international cohorts comprising 762,588 videos from 25,205 studies.

Image quality was a major determinant of AI performance across echocardiographic measurements, dysfunction and disease detection, view identification, and left-ventricular segmentation. High-quality studies achieved the best performance in 80% of evaluations, while a quality-aware aggregation strategy improved performance in 84% of evaluations. These findings show that incorporating image-quality awareness can improve the robustness and clinical reliability of echocardiographic AI systems.

---

> **Pretrained weights**: Available at 🤗 https://huggingface.co/CardiovascularImagingVHIR/EchoQuality

## Examples

Same patient, same study, same view, shown at three different quality levels.

<img src="content/quality_views_3x4.gif" alt="EchoQuality examples" width="100%">

## How to use

1. Download the file `echoqual.pt` from the HuggingFace link above.
2. Place it in: `results/echoQual_inference/`.
3. Create and activate the environment using the provided `content/torchone.yml` file.
4. Run the script with the directory containing your input videos as the `root_path` argument:
   ```bash
   python inference_echoqual.py /path/to/videos
   ```
   Input videos should have **256×256 pixel resolution** and **original frame length**, preferably in **MP4 or AVI format**.
5. The script generates two output files in `results/echoQual_inference/`:
   - `inference_sample_predictions_*.csv`: predictions at sample level.
   - `inference_predictions_*.csv`: aggregated predictions at video level.

6. In case of error while loading the backbone, download the file `backbone_structure.pt` from the HuggingFace and place it in: `results/echoQual_inference/`.

## Citation

If you use this work, please cite ... (not yet)
