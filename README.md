# Fake-Review-Detection
Detecting Fake Reviews using Semi-Supervised Learning from the Yelp Restaurant Reviews Dataset

## Directory Structure
- Code
  - main.py --> Main Python File containing the code for the entire project
  - FakeReviewDetection.sh --> Script File to Run main.py. ```(This file installs all the libraries required for running the project using pip3 and runs the code using python3. If using a different version, please change the command from pip3 to pip/pip2 and python3 to python depending on your version.)``` 

- Data
  - df.csv --> This is a cleaned, pre-processed and feature engineered version of the original dataset. (Provided for reference) 

- Evals
  --> Contains Screenshots of various outputs and test runs of both the ML models.

## How to Run

1. Clone the github repository onto your desktop.
2. Download the original dataset from the the following [link](https://drive.google.com/drive/folders/160kLuaEm-r8IUNHMPln6qK2Odjv7Lh1g?usp=sharing) (936MB). (Unable to upload this file as Github does not support uploading large files.)
3. Copy the dataset .db file in the following directory. ```Fake-Review-Detection/Data```
4. Open terminal inside the Code directory.
5. Enter the following command to give permissions to FakeReviewDetection.sh file. 
    ```chmod 777 FakeReviewDetection.sh```
6. Run the script file using the following command. ```./FakeReviewDetection.sh```
