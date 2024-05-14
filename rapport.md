## Stage 1 : Weights quantization

### _Record the range of the weights, as well as their 3-sigma range (the difference between μ−3σ and μ+3σ)_

Résultats :
```
n: 61770
min: -0.6351189017295837
max: 0.547119140625
3-sigma range: [-0.1706021090503782, 0.16470475145615637]
```

### _Explain which range you used for your quantization. Does range have an impact on model performance in this case ? Explain your answer._

On obtient une accuracy d'environ 50% tant avec un range `[-max, max]` qu'avec un range `[-3-sigma, 3-sigma]`.

Si l'on regarde les valeurs de poids min, max et 3-sigma de chaque couche:

```
min tensor(-0.5442, grad_fn=<MinBackward1>)
max tensor(0.6623, grad_fn=<MaxBackward1>)
3std tensor(0.5995, grad_fn=<MulBackward0>)

min tensor(-0.4676, grad_fn=<MinBackward1>)
max tensor(0.4881, grad_fn=<MaxBackward1>)
3std tensor(0.3177, grad_fn=<MulBackward0>)

min tensor(-0.2533, grad_fn=<MinBackward1>)
max tensor(0.3240, grad_fn=<MaxBackward1>)
3std tensor(0.1273, grad_fn=<MulBackward0>)

min tensor(-0.2388, grad_fn=<MinBackward1>)
max tensor(0.2291, grad_fn=<MaxBackward1>)
3std tensor(0.1938, grad_fn=<MulBackward0>)

min tensor(-0.4616, grad_fn=<MinBackward1>)
max tensor(0.3305, grad_fn=<MaxBackward1>)
3std tensor(0.3784, grad_fn=<MulBackward0>)
```

On constate que les disparités entre un range `[-max, max]` et `[-3-sigma, 3-sigma]`
restent mesurées. Si ce n'était pas le cas, on pourrait avoir des résultats différents où
l'un des deux ranges ne capturerait pas la majorité de la distribution des poids.

### _Do you observe a drop in the general accuracy ? If you did everything right, it should be negligible. Explain your findings._

Non, on obtient une accuracy d'environ 50% avec ou sans quantification.


### _Compare the memory footprint of the original model and the quantized one. Did the memory footprintchange ? Explain your findings. You can use torchinfo or torch-summary to get the memory footprint._

Non, il n'y a pas de changement car les poids sont toujours stockées à l'aide de `float`, même si
leurs valeurs sont compatibles avec une représentation en entier 8 bits.
