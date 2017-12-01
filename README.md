# 3D scattering transform

To install a fresh environment to run the code, do

```shell
conda create --name myenv --file spec-file.txt
```

## Running the 3D scattering transform

Example code is in `perform_scattering_on_dataset.py`. Image data should be
saved in a numpy-compatible format with dimensions `n_samples x width x height x depth`, where `width`, `height`
and `depth` are all powers of 2. Output gets saved automatically to generated file name.

## Classification

Example code is in `classify_ct_scans.py`.

## Checking the Littlewood-Paley condition

Example code is in `littlewood_paley_condition.py`. The filter bank is contructed in `filter_bank.py`.
Running `littlewood_paley_condition.py` gives output like:

```
making filter bank...
largest element in unnormalised raw lp sum:  1.08858525003
done in  5.235877513885498
LP condition satisfied:  True
with epsilon: 0.727835937593
average epsilon: 0.354274179652
bandwidth of mother wavelet: 141.1109510436776
bandwidth radians: 6.926767611560726
Complete lp sum:
[[[ 1.          0.27216406  0.69590791  0.27216406]
  [ 0.29313275  0.49666957  0.77198672  0.47346843]
  [ 0.81072562  0.85550399  0.89233088  0.85550399]
  [ 0.29313275  0.47346843  0.77198672  0.49666957]]

 [[ 0.27596148  0.48273337  0.77477524  0.47239091]
  [ 0.50685389  0.66206941  0.83047116  0.62957182]
  [ 0.8951128   0.91862351  0.89779179  0.90833465]
  [ 0.49794594  0.6313169   0.82385386  0.64447555]]

 [[ 0.57260778  0.58571186  0.58955019  0.58571186]
  [ 0.6128503   0.62934809  0.60089379  0.6048572 ]
  [ 0.69639007  0.67224511  0.62671177  0.67224511]
  [ 0.6128503   0.6048572   0.60089379  0.62934809]]

 [[ 0.27596148  0.47239091  0.77477524  0.48273337]
  [ 0.49794594  0.64447555  0.82385386  0.6313169 ]
  [ 0.8951128   0.90833465  0.89779179  0.91862351]
  [ 0.50685389  0.62957182  0.83047116  0.66206941]]]
```

This is after tuning `sigma` to get close to optimal in terms of how small `epsilon` is for a 128x128x128 filter bank.

## Visualisation

It is possible to get a heat map of original pixels that are important for the classification. Example code is in
`visualisation_gradient.py`
