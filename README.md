# Biased-Regression

A custom Torch criterion based on the [MSECriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.MSECriterion) to train a nueral network that will learn to either over or under predict. This may be useful in certain situations i.e when over predicting is financially more expensive than under predicting.  

The example trains a model using the [Boston Housing Data](http://lib.stat.cmu.edu/datasets/boston).

### Options
```lua
  -- The lower the bias weight the more the model will under / over predict.
  criterion_opt.biasWeight = 0.05
  criterion_opt.underPredict = true
  criterion = nn.BiasedMSECriterion(criterion_opt)
```  

### Issues
Although the model appears to work ok and the outputs are similar to whats expected but the gradCheck test is failing.
``` 
 th gradCheck.lua
```

### Under Predicting

``` 
 th -i main.lua --under
```


``` 
#   prediction     actual      diff	
 1     12.00       12.10      -0.10	
 2     19.04       27.50      -8.46	
 3     22.64       32.00      -9.36	
 4      5.82        8.80      -2.98	
 5     16.80       17.40      -0.60	
 6     19.92       21.70      -1.78	
 7     13.54       23.10      -9.56	
 8     22.64       30.10      -7.46	
 9     28.07       50.00     -21.93	
10     22.39       27.10      -4.71	
11     15.11       19.90      -4.79	
12     16.10       24.50      -8.40	
13     21.13       26.60      -5.47	
14     17.06       20.10      -3.04	
15     27.08       50.00     -22.92	
16     18.90       19.40      -0.50	
17     19.10       24.30      -5.20	
18     31.22       50.00     -18.78	
19     13.52       14.30      -0.78	
20     16.91       14.50       2.41
```

### Over Prediciting

``` 
 th -i main.lua --over
```

```
#   prediction     actual      diff	
 1     19.40       12.10       7.30	
 2     26.95       27.50      -0.55	
 3     34.34       32.00       2.34	
 4     20.53        8.80      11.73	
 5     25.07       17.40       7.67	
 6     25.44       21.70       3.74	
 7     25.97       23.10       2.87	
 8     36.05       30.10       5.95	
 9     50.17       50.00       0.17	
10     30.52       27.10       3.42	
11     22.51       19.90       2.61	
12     22.53       24.50      -1.97	
13     28.96       26.60       2.36	
14     23.30       20.10       3.20	
15     50.87       50.00       0.87	
16     25.82       19.40       6.42	
17     27.24       24.30       2.94	
18     46.33       50.00      -3.67	
19     21.28       14.30       6.98	
20     23.17       14.50       8.67
```

### Normal Regression
The example can also run normal regression using the standard Torch MSECriterion.

``` 
 th -i main.lua
```
