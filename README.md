# OCT Classicication task for AMD/DME/CNV/DRUSEN
## Requirements:
  * Python==3.11.8
  * Pytorch==2.2.1 + Cuda == 12.1
  * Tensorboard==2.16.2
  * Numpy==1.24.3
### A-Download the dataset from this following links:
  | Resource | download link |
  | -------- | -------- |
  | BOE    | [Click here to download](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm) |
  | CELL    | [Click here to download](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
  | RetinalOCT    | [Click here to download](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8) |
### B-Move these datasets to "OCT-Classifier/dataset" directory
  **1.**  Run the xx_dataset_split.py to split dataset automatically.
  
  **2.**  After splitting, dataset folder structure should be like following:
  #### Take the BOE as an example：
  - Semi_BOEdata
    - test
      - AMD
      - DME
      - NORMAL
    - train
      - AMD
      - DME
      - NORMAL
    - unlabel
      - AMD
      - DME
      - NORMAL
    - val
      - AMD
      - DME
      - NORMAL
### C-Run the model:
  **1.** Configure the "config.py", set the dataset name and the dataset root directory.

  **2.** Set the args in "train.py". 
   - "--cfg" refers to the "config.py" above.
   - "--out" refers to the output directory.

  **3.** Run the "train.py" script. You can check the training process on console or tensorboard, logs will be saved in the output directory that set in step 2, model parameters will be saved too.
 
