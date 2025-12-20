# Dynamic Feature Identification in Echocardiograms

This repository contains signal-processing-based algorithms for identifying
cardiac structures in echocardiogram data, including mitral valve leaflet
detection (MATLAB) and left ventricle identification (Python).

---

# Dynamic Feature Identification in Echocardiograms

This repository contains signal-processing-based algorithms for identifying
dynamic cardiac structures in echocardiogram data. The project focuses on
automatic detection of mitral valve leaflet motion and left ventricle geometry
using classical digital signal processing (DSP) and image analysis techniques,
without the use of learning-based models.

The implementation includes:
- A MATLAB-based pipeline for **mitral valve leaflet detection** using temporal
  motion energy and spatial filtering.
- A Python-based pipeline for **left ventricle identification** using image
  filtering, segmentation, and geometric heuristics.

This work was developed as a final project for a Digital Signal Processing
course at Columbia University.

---

## Repository Structure

DSP-Final-Project-2025-Echocardiograms/
├── run_me.m
├── echodsp_leaflet_demo1.m
├── echodsp_leaflet_demo2.m
├── left_ventricle_identification_alg.py
├── heart_images/
│   └── (echocardiogram image dataset)
├── references/
│   └── (papers, documentation, and reference material)
├── figures/
│   └── (output figures and visualizations)
└── README.md

## Requirements

### MATLAB
## MATLAB Dependencies

The following MATLAB toolboxes are required to run
`echodsp_leaflet_demo1.m`:

### Required Toolboxes

- **Image Processing Toolbox**
  - Functions used:
    - `imgaussfilt`
    - `rgb2gray`
    - `im2single`
    - `mat2gray`
    - `imshow`
    - `bwareaopen`
    - `imclose`
    - `strel`
    - `bwmorph`
    - `imdilate`

- **MATLAB (base installation)**
  - Functions used:
    - `VideoReader`
    - `VideoWriter`
    - `readFrame`
    - `conv2`
    - `zeros`, `size`, `mean`, `std`, `abs`
    - Plotting and figure utilities (`figure`, `subplot`, `title`)

### Python
- Python 3.8 or newer
- Required Python packages:
  - os
  - numpy
  - opencv-python
  - matplotlib
  - scipy
  - scikit-image
  - tkinter (included with standard Python installations)

## How to Run

This repository contains two independent algorithms:
1. A MATLAB-based mitral valve leaflet detection algorithm
2. A Python-based left ventricle identification algorithm

They can be run separately as described below.

---

### Mitral Valve Leaflet Detection (MATLAB)

The mitral valve leaflet detection algorithm is implemented in MATLAB and
operates on echocardiogram video data.

To run the MATLAB pipeline:

1. Open MATLAB
2. Set the current working directory to the root of this repository
3. Run the following command in the MATLAB Command Window:

```matlab
run_me.m
```

### Left Ventricle Identification (Python)
The Left Ventricle Identification algorithm is implemented in Python and
operates on echocardiogram image data.

To run the Python pipeline:

1. Download the references and heart_image folder
2. Open ```run_me.py``` Python program in your preferred IDE.
3. In your IDE terminal, change your directory to the folder. 
4. Run the ```run_me.py``` Python program, an LV diagram calculation will pop out. ~~which will cause a GUI to appear~~
~~3. For the "Heart Images Folder Path", input the full path to the heart_images folder~~
~~4. For the "Save Folder Path", input the full path to the folder you want to save the images to~~
~~5. For the "Reference Ventricle Path", input the full path to reference_ventricle.jpg, which is provided to you in the references folder~~
~~6. For the "Reference Background Path", input the full path to the reference_background.npy, which is provided to you in the references folder~~
~~7. Press Go to run the algorithm~~
~~8. Check the save folder to see that the images are uploaded~~

