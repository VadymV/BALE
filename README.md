# BALE (Bimodal Affective LEarning)
This repository contains the code for the paper *Decoding affective states without labels: bimodal image-brain supervision* submitted to *ACM International Conference on Multimodal Interaction*. 

Further details will be uploaded on May 4.

## Install dependencies

``uv`` is used as package manager. To install the dependencies, run:
```py
uv sync
git clone https://github.com/Cognitive-Computing-Group/NEMO.git
git clone https://github.com/Kallemakela/mne-bids.git
uv pip install -e external/nemo
```

## Download and prepare NEMO
1. Download *nemo-bids.zip* from https://osf.io/pd9rv/
2. Extract into the folder `dat/nemo-bids/`
3. `git clone https://github.com/Cognitive-Computing-Group/NEMO.git`
4. `cd NEMO`
5. `conda env create -f environment.yaml -n nemo`
6. `conda activate nemo`
7. `cd ..`
8. `python convert_bids_to_fif.py` to convert the BIDS dataset into FIF format. The script will create a folder `data/nemo-preprocessed-data/` with the converted data.
9. `conda deactivate`
10. Extracting embeddings for images: TODO

## Download and prepare AOMIC
1. TODO

## Run the training
```py
uv sync
uv run python .\run_experiments.py -d dataset -cv subject-dependent
```
where dataset is one of the following: `nemo` or `aomic` and
cv is one of the following: `subject-dependent` or `subject-independent`

## Generate the results and the figures
```py
uv run python generate_results_figures.py
```