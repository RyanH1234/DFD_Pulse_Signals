# rPPG w. Motion Detection

## To Run

- Download the DFDC challenge dataset from https://www.kaggle.com/c/deepfake-detection-challenge
- Replace line 588 in `run.py` with the relevant path to the DFDC dataset
- Run the following command
```
python run.py
```
- The above command may take a while to run.
- Once it has completed run navigate to `model.py`
- In the main function there are a number of commented out functions e.g. `run_SVM()` etc..
- Uncomment <b> one </b> of these methods
- Run the following command
```
python model.py
```