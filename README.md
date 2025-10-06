# Sentinel-2 Cloud Removal using Conditional GANs (Pix2Pix)

This repository contains the final project for the **AI4EO Platform and Best Practices** course. The project implements a deep learning pipeline using a Conditional Generative Adversarial Network (cGAN), specifically the Pix2Pix model, to remove clouds from optical Sentinel-2 satellite imagery.


*An example of a (Cloudy, Clear) image pair used for training the model.*

## Project Overview

Cloud cover is a significant limitation in Earth Observation (EO), obscuring land surfaces and rendering large volumes of optical satellite data unusable for downstream analysis. This project tackles this fundamental challenge by training a generative model to "inpaint" or "hallucinate" plausible ground features underneath clouds.

The core of this project is a Pix2Pix model, a cGAN architecture designed for image-to-image translation tasks. The model learns a mapping from cloudy Sentinel-2 images to their corresponding clear-sky counterparts. A key aspect of this project is the **self-supervised data acquisition strategy**, where (cloudy, clear) image pairs are automatically generated from the Sentinel-2 archive on Google Earth Engine (GEE), eliminating the need for manual annotation.

## Results: From Mid-term to Final

The project underwent iterative refinement based on mid-term feedback. The initial results suffered from common GAN training issues such as checkerboard artifacts, `nodata` contamination, and unstable training dynamics. After improving data preprocessing, paired normalization, and fine-tuning the Pix2Pix model, the final results show a substantial improvement in both visual quality and quantitative metrics.

### Qualitative Results

Visual inspection demonstrates that the final Pix2Pix model effectively removes clouds and reconstructs plausible land surface features. Both thin and thick cloud regions are addressed, and generated landscapes preserve structural patterns such as field boundaries, vegetation patches, and water bodies.

* **Thick clouds:** Generated textures are slightly smoother, but overall land cover patterns are reasonable.
* **Thin clouds:** High-frequency details, such as edges and small features, are well-preserved.
* **Color consistency:** Generated images maintain realistic colors and brightness relative to the ground truth.

| Cloudy Input | Generated Output | Ground Truth |
| :---: | :---: | :---: |
| ![Cloudy](images/cloudy_sample.png) | ![Generated](images/generated_sample.png) | ![Clear](images/clear_sample.png) |

### Quantitative Results

The model performance is evaluated using **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)**, computed only over valid (non-`nodata`) pixels to avoid bias.

| Metric | Test Set Average | Standard Deviation |
| :--- | :---: | :---: |
| PSNR (dB) | 22.59 | 6.80 |
| SSIM | 0.59 | 0.24 |

* **PSNR:** Indicates moderate fidelity to the ground truth. Higher PSNR corresponds to better reconstruction of obscured regions. The high standard deviation reflects variability across images, particularly those with very dense clouds.
* **SSIM:** Confirms structural similarity preservation. Dense cloud regions reduce fine-detail reconstruction, which is expected in this ill-posed problem.

### Diagnostic Analysis

* The histograms show that generated images match the overall brightness and color distribution of ground truth images.
* Slight smoothing occurs in areas fully obscured by clouds, consistent with the model’s inability to perfectly infer missing details.
* Early versions suffered from dark outputs or compressed dynamic ranges, corrected by paired normalization and cropping to common valid regions.

![Histogram Comparison](images/figure3.png)  
*Histogram comparison between generated outputs (center) and ground truth (right) showing reasonable spectral reconstruction.*

### Summary of Improvements

| Aspect | Initial Model | Final Model |
| :--- | :---: | :---: |
| Cloud Removal Quality | Artifacts, incomplete removal | Visually plausible, cloud-free output |
| Structural Preservation | Poor, blurred edges | Clear field boundaries and textures |
| Color & Brightness | Inconsistent | Correct and realistic |
| Training Stability | Unstable, slow convergence | Stable, smooth learning curve |

* **Key takeaway:** The combination of robust preprocessing, paired normalization, and careful tuning of Pix2Pix hyperparameters allows the model to produce realistic cloud-free images, providing a strong foundation for downstream tasks like land cover classification or change detection.

*(Note: These are preliminary results from a 10-epoch run. The final report will include results from the full 200-epoch training.)*

## How to Reproduce the Results

Follow these steps to set up the environment, process the data, and train the model.

### Prerequisites
*   Git
*   Conda package manager
*   A Google Earth Engine account

### Step 1: Installation

First, clone this repository and the required Pix2Pix implementation. Then, create the Conda environment using the provided file.

```bash
# Clone this project repository
git clone <your_repository_url>
cd <your_repository_name>

# The training code is based on the pytorch-CycleGAN-and-pix2pix repository
git clone https://github.com/junyanza/pytorch-CycleGAN-and-pix2pix.git

# Create and activate the conda environment
cd pytorch-CycleGAN-and-pix2pix
conda env create -f environment.yml
conda activate pytorch-img2img
cd ..
```

### Step 2: Data Acquisition (Google Earth Engine)

The training data is not included in this repository. You must download it yourself using Google Earth Engine.

1.  Log in to the [GEE Code Editor](https://code.earthengine.google.com/).
2.  Copy the contents of the `gee_data_downloader.js` script from this repository into the editor.
3.  Adjust the `ROI` (Region of Interest) and `MAX_PAIRS_TO_EXPORT` variables if desired.
4.  Run the script. This will start multiple export tasks to your Google Drive.
5.  Download all the exported `.tif` files from your Google Drive into a folder named `cloud_removal_dataset_california` in the root of this project directory. The final structure should be:
    ```
    .
    ├── cloud_removal_dataset_california/
    │   ├── clear_0.tif
    │   ├── cloudy_0.tif
    │   ├── clear_1.tif
    │   ├── cloudy_1.tif
    │   └── ...
    ├── preprocessing.ipynb
    └── train.ipynb
    ```

### Step 3: Data Preprocessing

Run the `preprocessing.ipynb` notebook. This script will:
1.  Validate the downloaded image pairs.
2.  Crop them to their common valid `nodata` region.
3.  Apply a consistent normalization to each pair.
4.  Split the data into `train`, `val`, and `test` sets.
5.  Save the processed, concatenated PNG files into the `dataset/` directory.

**Note:** This notebook is designed to be run once. If the `dataset/` directory already exists, it will skip preprocessing.

### Step 4: Training and Evaluation

Run the `train.ipynb` notebook. This notebook will:
1.  Set the training parameters (e.g., epochs, batch size).
2.  Execute the training script from the `pytorch-CycleGAN-and-pix2pix` repository.
3.  Generate results on the test set.
4.  Calculate PSNR and SSIM metrics on the test results.
5.  Visualize a sample of the outputs.

## Addressing Mid-term Feedback

This project was significantly improved by addressing the feedback from the mid-term report.

> **1. Introduction should be greatly expanded... Missing discussion of prior work.**
>
> This has been addressed in the final written report with an expanded literature review section.

> **2. How confident are you that this ± 45 day window does not result in seasonal changes?**
>
> This is a valid concern. While the GEE script can be adjusted to a shorter window, the primary mitigation was a **robust data preprocessing pipeline**. The new `preprocessing.ipynb` now crops images to their common valid data region and applies a *paired normalization* strategy, ensuring that the brightness and contrast scaling is identical for both the cloudy and clear images in a pair. This minimizes the impact of minor seasonal variations in illumination.

> **3. Model choice makes sense. Would love to see comparisons of all 3 models mentioned here.**
>
> Due to time constraints and the complexity of GAN training, this project focused on deeply optimizing the Pix2Pix model. A comparison with CycleGAN and other architectures is discussed as a key area for future work in the final report.

> **4. Honestly, results don't look great so far... Why does the generated cloud-free output have black pixels?**
>
> This was the most critical feedback. The poor initial results were caused by two main issues, which have been fixed:
> *   **`Nodata` Contamination:** The `preprocessing.ipynb` script now intelligently finds the common `nodata` mask between a cloudy/clear pair and crops to that specific region, completely eliminating black `nodata` pixels from the training data.
> *   **Faulty Normalization:** The initial per-image normalization was replaced with a paired normalization strategy, which was critical for the model to learn the correct mapping.

> **5. I'm also very interested in whether these generated cloud-free images can enhance predictive performance on a benchmark dataset.**
>
> This is an excellent suggestion for a follow-on project. The final report includes a "Future Work" section detailing how these generated images could be used to improve downstream tasks like land cover classification, serving as a form of data augmentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
*   The core model implementation is based on the excellent [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository by Zhu et al.
*   Satellite imagery was sourced from the Copernicus Sentinel-2 program.
