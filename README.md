# Deep Learning to Analyze Sliding Drops ([Cite Us](#citation))

The **SFOF4S toolkit** is designed to solve existing challenges in analyzing the sliding motion of drops on tilted plates. It enables researchers to conduct detailed studies by considering a wide range of variables and their correlations.

Key features include:
- **Super-Resolution Enhancement**: The toolkit enlarges images by an up-scale ratio of 3 using a super-resolution model trained on a dataset of 14,000 sliding drop images.
- **Comprehensive Analysis**: Extracts various criteria from the super-resolution images to provide precise insights into drop behavior.
- **Optimized Contact Angle Measurement**: Contact angles are measured using an optimized polynomial method, as described in the accompanying manuscript.

This toolkit streamlines complex image processing tasks, empowering researchers with high-quality data for in-depth analysis.

![graphical abstract](https://user-images.githubusercontent.com/57271994/194614949-8736973c-4df3-4449-9a21-bc2423405648.png)

---
# General inforamtion

The original implementation had dependency issues, so a new version was introduced to resolve them by migrating to the PyTorch framework.
Please note that installing this package requires CUDA libraries and may occupy up to 3 GB of storage.
Check a simple implementation ["https://github.com/yriyazi/SFOF4S-Documentation]("https://github.com/yriyazi/SFOF4S-Documentation)

## Drop Profile Video

- **File:** `drop_profile.mp4`

    This video illustrates a sliding drop on a sample with a defect in the middle:

    - The **first drop** in the upper part of the video is the real drop image after preprocessing. Key preprocessing steps include:
      - Calculating the tilt angle and aligning frames horizontally.
      - Removing noise and background.
      - Detecting the drop position (marked by red lines).
    - The **second drop** (larger image) represents the drop after applying a super-resolution model.
    - Below that, the **drop contour** is extracted, and parameters such as contact angles (CAs), drop height, and drop length are displayed.
    - On the left, four plots analyze how the drop slides over a defected sample.

[Preview the video on GitHub](https://github.com/AK-Berger/4S-SROF/assets/57271994/4130c91c-193a-4bf9-b08d-e2df878f88bf)

---

## Instructions

- **File:** `instructions.md`  

    This file is a comprehensive guide designed to assist researchers in setting up and using the analysis toolkit. It has two general sections: one focused on setting up the environment and another on effectively using the code.

    - **How to Set Up:**  
      This section guides you through the installation of Python, Anaconda, and Jupyter Notebook. It explains how to download the code repository, prepare your working directory, and ensure all necessary libraries are installed for a seamless setup.  

    - **How to Use:**  
      This section provides detailed steps for preparing video frames, creating and marking the slope folder, and running the analysis code. It also explains how to configure experiment-specific variables and interpret the generated outputs, such as `result.xlsx` and analyzed frame outputs.

---
## Control Panel

- **File:** `Control Panel.ipynb`  

    This file contains the main executable code for the toolkit. It acts as a central hub, calling other `.py` files and libraries to perform the analysis. The code processes the video frames, extracts features, and generates the final `result.xlsx` file containing the analyzed time-series data. It integrates all steps of the workflow, making it easy to convert raw video frames into meaningful results.

---

## Tutorial

- **tutorial:** `tutorial.ipynb`  and dataset
    [HERE](https://github.com/yriyazi/SFOF4S-Documentation)
    

---


## Dataset

- **Description:** The dataset contains 14,000 sliding drop images used to train the modified super-resolution model.
- **Access:** [Download the dataset](https://www.kaggle.com/datasets/sajjdeus/4s-srof)

---

## ESPCN Super-Resolution Model Weights

- **File:** `SuperRes_weights.h5`

    This file contains the weights for the modified ESPCN super-resolution model, trained on 14,000 sliding drop images.

---

## Python Files

The following `.py` files are included in the toolkit and provide functions for related image processing tasks:

- `angle_detection.py`
- `baseline_detection.py`
- `criteria_definition.py`
- `edge_superres.py`
- `tools.py`
- `visualization.py`

These scripts are essential for researchers working with this toolkit.


---
# Citation

If you find **"Deep Learning to Analyze Sliding Drops"** useful for your research, please consider citing the paper using the following information:

<div style="position: relative;">
  <pre>
    <code>
@article{shumaly2023deep,
  title={Deep learning to analyze sliding drops},
  author={Shumaly, Sajjad and Darvish, Fahimeh and Li, Xiaomei and Saal, Alexander and Hinduja, Chirag and Steffen, Werner and Kukharenko, Oleksandra and Butt, Hans-J{\"u}rgen and Berger, R{\"u}diger},
  journal={Langmuir},
  volume={39},
  number={3},
  pages={1111--1122},
  year={2023},
  publisher={ACS Publications}
}
    </code>
  </pre>
  <button onclick="copyText(this)" style="position: absolute; top: 0; right: 0;"></button>
</div>





---
