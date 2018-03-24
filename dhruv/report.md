###
./runs/1517232217/checkpoints/ -> trained only on books, 2300 steps
./runs/1517407573/checkpoints/ -> trained only on dvd, 1300 steps
./runs/1517419772/checkpoints/ -> trained only on music, around 2200 steps

### Accuracy
rows mean that a given dataset is evaluated on different models.

columns mean that a given model is evaluated on different data sets

|models->|books|dvd|music|
|---|---|---|---|
|books|0.766|0.7325|0.6815|
|dvd|0.7165|0.756|0.7105|
|music|0.709|0.7455|0.77|

### Experiments

#### Failed attempts:
the following configurations did not work, accuracy below 60%
Loss function eqn:
- min(s + d)
- min(s - d)
- min(s + 1/d)
- jointly minimising s and 1/d

#### Successful attempts:
accuracy is close to baseline model.

perform all possible iterations for the following configurations (i.e. all domain pairs and individual research questions):

##### min(s + alpha/d)
- alpha = 0.01, 0.05, 0.1, 0.2,

##### randomly train min(s) and min(1/d) in the ratio:
- min(1/d) frequency = 0.05, 0.01, 0.1, 0.2, 0.3

##### do both of the above at the same time
- alpha, frequency = (0.01, 0.01), (0.1, 0.1), (0.2, 0.2),
