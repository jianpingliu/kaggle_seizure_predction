## Kaggle competition: Melbourne University AES/MathWorks/NIH Seizure Prediction

This is simple solution to the [Kaggle competitoin: Seizure Prediction](https://www.kaggle.com/c/melbourne-university-seizure-prediction). The solution is a Tensorflow implementation of a similar solution based on [neon](https://github.com/anlthms/sp-2016).  

### Train model

Here is the command to train the model:

```
python train.py \
--classes=2 \
--train_option=train \
--train_file=train_1 \
--model_name=train_1 \
--lr=1e-3 \
--batch_size=32 \
--num_epochs=20 \
--checkpoint_every=20 \
--evaluate_every=20 \
--train_dev_split=0.20
```

### Predict test data

Command to predict test data:

```
python predictions.py \
--model_name=train_1 \
--output_file=test_1.csv \
--test_file=test_1
```