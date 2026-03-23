### Load Saturation Detection

This repository accompanies the article "A Framework for Detecting The Onset of Load Saturation in Capacity-Constrained Systems" submitted to IEEE Access, manuscript ID: Access-2026-12416

The repository contains:
- [implementation](./pyloadsat) of the proposed Variance Response–Based Saturation Detection (VRSD) framework in Python
- synthetic data as NumPy [binary files](./pyloadsat/data)
- interactive Jupyter [notebook](./synthetic_data.ipynb) (same on [nbviewer.org](https://nbviewer.org/github/andrewha/load_saturation_detection/blob/main/synthetic_data.ipynb)) for reproducing saturation effect
- standalone Python [script](./synthetic_data.py)

Due to confidentiality constraints, real operational data for legacy voice network cannot be shared.

---
It is recommended to run the code in a virtual Python environment. Python 3.10 or later is required.
##### Clone repository
```
> mkdir load_saturation_detection
> git clone https://github.com/andrewha/load_saturation_detection.git
> cd load_saturation_detection
```
##### Create and activate your Python virtual environment, for example, like this
```
> \path\to\python\python.exe -m venv myenv
> myenv\Scripts\activate
```
##### Install dependencies in your Python virtual environment
```
> pip install -r requirements.txt
```
##### Run interactively
Open and run the `synthetic_data.ipynb` notebook in Jupyter or IDE supporting Jupyter notebooks.
##### Run end-to-end
Run the standalone non-interactive `synthetic_data.py` script producing the same results:
```
> python synthetic_data.py
```

---
Copyright (c) 2026 Andrei Batyrov

[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD-blue)](https://opensource.org/licenses/BSD-3-Clause)
