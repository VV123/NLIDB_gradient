[NLIDB](../../README.md) | [Annotation](README.md)

## Adversarial Text Method
      
- Preprocess data for *annotation* and *binary classifier*
  ```
  utils/end2end_anno/prep_files.py
  ```
- Build a *binary classifier* and produce gradient norms of each token

  ```
  utils/end2end_anno/wc_conv1.py
  ```
- Pinpoint mentions using gradient norms from the previous step

  ```
  utils/end2end_anno/pin_adversarial.py
  ```
