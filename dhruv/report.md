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
