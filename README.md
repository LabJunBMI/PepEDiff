# PepEDiff
This is the implementation code for PepEDiff


## Reproducing Results

### 1. Testing Set & TIGIT
To reproduce the testing results you can download the processed data and model weight directly from:
[Here](https://zenodo.org/records/19521740)
  1. Run the sampling script ```python sample_testing.py``` to sample binders from testing set
  2. Execute all cells in ```sampl_TIGIT.ipynb``` to sample binders for TIGIT

### 2. Training
To train the model from scratch using the training dataset:

0.  **Download the data:**
    1. Download the structure data from [BioLip](https://aideepmed.com/BioLiP/)
    2. Place these files into the `/data` folder within this directory.

2.  **Preprocess the data:**
    1. Update the structure folder path in the notebook to match the location of the files downloaded in Step 0.
    2. Execute all cells in ```./data/preprocessing.ipynb```

3.  **Run the training script:**
    ```python train.py```




## Acknowledgments
This project builds upon the following open-source contributions:

* **Feature Extraction:** Derived from [biomed-AI/GraphEC](https://github.com/biomed-AI/GraphEC).
* **EGNN Architecture:** Based on the implementation by [vgsatorras/egnn](https://github.com/vgsatorras/egnn).

We sincerely thank the authors for their valuable contributions to the research community.
