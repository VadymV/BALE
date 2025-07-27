# BALE (Bimodal Affective LEarning)
This repository contains the code for the paper *Decoding Affective States without Labels: Bimodal Image-brain Supervision* accepted to *ACM International Conference on Multimodal Interaction* 2025, Canberra, Australia. 

## Download and prepare NEMO
1. Download *nemo-bids.zip* from https://osf.io/pd9rv/
2. Extract into the folder `dat/nemo-bids/`
3. `git clone https://github.com/Cognitive-Computing-Group/NEMO.git`
4. `cd NEMO`
5. `conda env create -f environment.yaml -n nemo`
6. `conda activate nemo`
7. `cd ..`
8. `python convert_nemo_bids_to_fif.py` to convert the BIDS dataset into FIF format. The script will create a folder `data/nemo-preprocessed-data/` with the converted data.
9. `conda deactivate`
10. `uv sync`
11. Set the correct paths in the config.yaml file (Lines: 46 - 52).
11. `uv run python .\extract_nemo_features.py`.
12. Download IAPS images (https://psycnet.apa.org/record/2007-08864-002; https://osf.io/d8sru) and place them in `data/nemo/images/` such that this folder contains only jpg images.

## Download and prepare AOMIC (request for access is required)
1. AOMIC dataset requires ADFEC dataset (https://aice.uva.nl/research-tools/adfes-stimulus-set/adfes-stimulus-set.html?cb)
2. Download the videos, extract them and run the script `extract_aomic_frames.py` to convert the videos into frames.
3. `uv run python .\copy_last_aomic_frame.py`
4. Download DiFuMo data from https://figshare.com/projects/Self-Supervised_Learning_of_Brain_Dynamics_from_Broad_Neuroimaging_Data/172176
5. Locate `ds002785` extract the downloaded data and run the script `extract_aomic_recordings.py`

## Extract image features from images
`uv run python .\extract_image_features.py`

## Run the training
```py
uv run python .\run_experiments.py -d dataset -cv strategy
```
where dataset is one of the following: `nemo` or `aomic` and
strategy is one of the following: `subject-dependent` or `subject-independent` 

## Generate the results and the figures
```py
uv run python generate_results_figures.py
```
