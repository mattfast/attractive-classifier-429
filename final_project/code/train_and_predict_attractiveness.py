import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from load_LFW_attributes import load_LFW_attributes, load_LFW_images

CUTOFF = 0
NUM_FOLDS = 4
LOSS_FUNC = 'binary_crossentropy'
BATCH_SIZE = 125
NUM_EPOCHS = 25

# Images are of size (250, 250, 3)
images = load_LFW_images('../lfw/')
print("Finished loading images")

attr = load_LFW_attributes('../lfw_attributes.txt')
print("Finished loading attributes")

y = []
for index, person in enumerate(attr):
    if person['Male'] > 0:
        y.append(person['Attractive Man'])
    else:
        y.append(person['Attractive Woman'])

print("Finished separating gender images")
images = np.array(images)

#encoder = LabelEncoder()
#encoder.fit(y)
#targets = encoder.transform(y)
targets = y
print('Finished Transforming Variables')

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
input_shape = images[0].shape
fold_no = 1

print('Starting model training')

best_acc = 0
accuracy = []
loss = []

# CNN Architecture from:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014093/
for train, test in kfold.split(images, targets):
    model = models.Sequential()
    model.add(layers.Conv2D(96, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(65, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='linear'))
    model.summary()

    model.compile(loss=LOSS_FUNC, optimizer='adam', metrics=['accuracy'])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(images[train], targets[train],
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              verbose=1)

    # Generate generalization metrics
    scores = model.evaluate(images[test], targets[test], verbose=1)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    if scores[1] * 100 > best_acc:
        best_acc = scores[1]*100
        model.save("../non_binary_dropout_model")

    accuracy.append(scores[1] * 100)
    loss.append(scores[0])

    fold_no += 1

# Generate summary data
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(accuracy)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss[i]} - Accuracy: {loss[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(accuracy)} (+- {np.std(accuracy)})')
print(f'> Loss: {np.mean(accuracy)}')
print('------------------------------------------------------------------------')









