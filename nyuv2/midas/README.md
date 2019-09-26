### MiDaS Evaluation

1. Download and unzip the NYU Depth v2 dataset from the link below and place the whole folder
in the data directory (so the directory tree should look like
 ```data/nyu_depth_v2_labeled_numpy/<file>.npy```)
 https://drive.google.com/file/d/1eqQCcUXXAl1zTZTtu8ao9RcG2us26WAD/view?usp=sharing
 
2. Put the MiDaS model.pt file in the `````./MiDaS````` directory as usual.
3. Create a conda environment using ```$ conda env create -f environment.yml```. Activate it.
4. Run the evaluation script by running ```$ python evaluate.py```. The console should spit out a lot of text
but at the end should give a summary of the average metrics on the dataset.

Mark Nishimura 9/24/19
