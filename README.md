Dear Vincent,

To install a fresh environment to run the code, do

```shell
conda create --name myenv --file spec-file.txt
```

The important code is in `filter_bank.py` and I check if the Littlewood-Paley condition
is satisfied in `littlewood_paley_condition.py`. Running the latter file gives output like:

```
making filter bank...
largest element in unnormalised raw lp sum:  1.00281314826
done in  1.6304457187652588
LP condition satisfied:  True
with epsilon: 0.367883010227
average epsilon: 0.193319292111
bandwidth of mother wavelet: 18.237778995787462
bandwidth radians: 7.1619590638700155
Complete lp sum:
[[[ 1.          0.75839997]
  [ 0.878622    0.99719474]]

 [[ 0.63211699  0.67441756]
  [ 0.78879604  0.72389835]]]
```

This the best I manage (in terms of how small `epsilon` is) for a 16x16x16 filter bank.
