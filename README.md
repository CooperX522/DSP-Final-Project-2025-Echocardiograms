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

They can be run separately by simply downloading the repository, and running the `run_me.m` and `run_me.py` files individually.

