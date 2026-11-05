# Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements

This repository contains the code for the ICRA 2026 paper ["Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements"](https://arxiv.org/abs/2602.03209).
A demo of the network can be seen in [this video](https://youtu.be/ES8my_vpLog?si=WkJuahyi9vYFSCVF).
The code enables to reproduce the evaluation results of our method and serve as a starting point for extensions and further research.
This work has been a collaboration between the [Autonomous Systems Lab at ETH Zurich](https://asl.ethz.ch/) and the [Field Robotics Lab at NTNU](https://www.ntnu.edu/frl).

## API
```python
from depth_completion import DepthCompletionModel
model = DepthCompletionModel()
metric_depth = model.predict(rgb_image, prior_points, intrinsics)
```

Inputs:
- `rgb_image`: NumPy array with shape `(H, W, 3)`
- `prior_points`: NumPy array with shape `(3, N)` containing sparse, metric 3D points in the camera frame
- `intrinsics`: NumPy array with shape `(3, 3)`

Output:
- `metric_depth`: NumPy array with shape `(H, W)` in meters

## Reproduce the results
```bash
./reproduce_results.sh
```
The script extracts the dataset, creates a local virtual environment, installs the required packages, runs the evaluation, and prints the summary line with the final metrics.

## Citation
```bibtex
@inproceedings{job2026depth,
    title={Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements},
    author={Marco Job and Thomas Stastny and Eleni Kelasidi and Roland Siegwart and Michael Pantic},
    booktitle={IEEE International Conference on Robotics and Automation},
    year={2026},
}
```

## Dataset
The dataset is stored in dataset/*.zip as Git LFS files. Make sure to install LFS support (e.g. `apt install git-lfs`) and then pull the files after cloning the repository (`git lfs pull`).

### File Structure
The archive has the following file structure:
```
|-- checkpoints             # Trained network weights
    |-- *.pth               # Pytorch model file
|-- training                # Datasets used for training
    |-- mountain_area
    |-- rhone2
    |-- road_corridor
    |-- rural_area
|-- evaluation              # Datasets used for validation
    |-- ballasttank
    |-- farmfield
    |-- glacierflight
    |-- industrialhall
    |-- underwaterfjord
|-- LICENSE.md              # Global license, with sublicense information
|-- README.md               # General information
|-- VERSION.md              # Versioning information
```

### Dataset Structure
Each dataset has the following structure:
```
|-- depth                   # Depth ground truth directory
    |-- 00000_d.npy         # Matrices containing metric depth values
    |-- 00001_d.npy
    |-- ....._d.npy
|-- depth_prior             # Optional: Depth measurements
    |-- 00000_rp.npy        # Matrices containing metric depth measurements in camera frame
    |-- 00001_rp.npy
    |-- ....._rp.npy
|-- rgb                     # RGB camera views
    |-- 00000_rgb.jpg
    |-- 00001_rgb.jpg
    |-- ....._rgb.jpg
|-- intrinsics.txt          # File containing poses of each camera frame in an inertial frame
|-- LICENSE.md              # Optional: License file specific to this directory
```

### Contact
Contact [Marco Job](https://www.ntnu.edu/employees/marco.job) or [Eleni Kelasidi](https://www.ntnu.edu/employees/eleni.kelasidi) for questions.
