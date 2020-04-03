[NLIDB](README.md) | [Annotation](utils/end2end_anno/README.md)

# NLIDB
A seq2seq NLIDB model augmented with adversarial machine learning
   
## Datasets
      
   [WikiSQL](https://github.com/salesforce/WikiSQL) [Annotated WikiSQL](https://drive.google.com/open?id=1fhW0_1Mvvg0xkGp3iPEfKU8xcC2ALPAk)
   
   [OVERNIGHT](https://worksheets.codalab.org/worksheets/0x269ef752f8c344a28383240f7bb2be9c)
   
   [ParaphraseBench](https://github.com/DataManagementLab/ParaphraseBench)
   
   [Spider](https://yale-lily.github.io/spider)

## Build Data
   Rebuild vocabulary and dataset 
   ```
   cd utils
   python data_manager.py
   ```
  
## Model

   ```
   USAGE
      $ python main.py
   OPTIONS
      --mode  [train, infer, transfer]
   ```
   
   Pre-trained [model](https://drive.google.com/open?id=1bVmRwTxl1-SQqHIJDh50og3tGcb3MGLB)



Feel free to drop [me](mailto:wenluwang@auburn.edu) an email or open an issue if you have any questions.
