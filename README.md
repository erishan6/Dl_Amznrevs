## Sentiment Analysis of Amazon Product Reviews Using Domain Adversarial Training

The dataset has been obtained from: https://www.uni-weimar.de/en/media/chairs/computer-science-and-media/webis/corpora/corpus-webis-cls-10/

 ## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy 

## Training

Print parameters:

```bash
./run.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement
  --source_data
                        Source data for training (default: books)
  --target_data
                        Target data for training (default: music)
  --domain_loss_factor_propagation
                        domain_loss_factor_propagation for training the loss_domain
  --domain_train_frequency
                        domain training frequency for training the loss_domain. A negative value implies seperate training is switched off
  --use_adam
                        Select optimizer to use. Default is AdamOptimizer, else use RMSPropOptimizer
  --activation_function
                        Select activation function to use. Default is relu

```

Train:

```bash
./run.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1521966686/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

## References

- Prettenhofer, Peter and Stein, Benno 2010. Cross- language Text Classification Using Structural Cor- respondence Learning. Proceedings of the 48th An- nual Meeting of the Association for Computational Linguistics.
- Xilun Chen and Ben Athiwaratkun and Yu Sun and Kil- ian Q. Weinberger and Claire Cardie. 2016. Adver-
sarial Deep Averaging Networks for Cross-Lingual Sentiment Classification. CoRR.
- Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Franc ̧ois and Marchand, Mario and Lempitsky, Victor. 2016. Domain-adversarial Training of Neural Networks, 17(1):2096–2030. J. Mach. Learn. Res.
- Denny Britz. 2015. Implementing a CNN for Text Classification in TensorFlow. http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.
- Denny Britz. 2015. dennybritz/cnn-text-classification- tf. https://github.com/dennybritz/cnn-text-classification-tf.

