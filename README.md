# Project 1: <span style="color:blue">Deadline 13 June 2024 @ 11:59PM</span>

---

## Team Members

| Name                 | Email Addess       | Student ID |
|:--------------------:|:------------------:|:----------:|
|Bryan Wong Hong Liang |bryanwong@u.nus.edu | A0215114X  |
|James Lim             | e0950510@u.nus.edu | A0251506M  |

---

# Deliverables 

- In this submission folder, we have included a few files with remarks as follows:
    - `distcomp.yaml`: Contains the consolidated package names and versions to install within an enclosed conda environment.
    - `housing_with_header.csv`: This file is the same as `housing.csv` file in Canvas with addition of column headers as 1st row.
    - `README.md`: This current document providing instructions on how to run the MPI pipeline.
    - `script.py`: The MPI pipeline with Python binding, with which Kernel Ridge Regression will be implemented in distributed manner within MPI environment
    - `summary.pptx`: Presentation slide deck that provides narration of the step-by-step implementation rationales, HyperParameter tuning details, and best RMSE result.

---

# Instructions to Run the MPI Pipeline

## Step 1: Set up Conda Environment

- Create a new conda environment called `distcomp` based on packages that are dumped into YAML file below.
- Run below command in the terminal.

```bash
conda env create --file distcomp.yaml
```

## Step 2: Activate `distcomp` Conda Environment

- Run below command in the terminal.

```bash
conda activate distcomp
```

## Step 3: Run MPI Pipeline with Python Binding

- The number of participating processors can be adjusted accordingly by replacing the argument `2` below.
- Please ensure to run below command at same directory level as `housing_with_header.tsv` file.
- Run below command in the terminal. 

```bash
mpiexec -n 2 python script.py
```
