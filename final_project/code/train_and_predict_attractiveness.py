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
NUM_EPOCHS = 15

# Images are of size (250, 250, 3)
images = load_LFW_images('../lfw/')
male_images = []
female_images = []
print("Finished loading images")

attr = load_LFW_attributes('../lfw_attributes.txt')
print("Finished loading attributes")

male_y = []
female_y = []
for index, person in enumerate(attr):
    if person['Male'] > 0:
        male_images.append(images[index])
        if person['Attractive Man'] > CUTOFF:
            male_y.append(1)
        else:
            male_y.append(0)
    else:
        female_images.append(images[index])
        if person['Attractive Woman'] > CUTOFF:
            female_y.append(1)
        else:
            female_y.append(0)

print("Finished separating gender images")
male_images = np.array(male_images)
female_images = np.array(female_images)

encoder = LabelEncoder()
encoder.fit(male_y)
male_targets = encoder.transform(male_y)
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
for train, test in kfold.split(male_images, male_targets):
    model = models.Sequential()
    model.add(layers.Conv2D(96, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss=LOSS_FUNC, optimizer='adam', metrics=['accuracy'])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(male_images[train], male_targets[train],
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              verbose=1)

    # Generate generalization metrics
    scores = model.evaluate(male_images[test], male_targets[test], verbose=1)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    if scores[1] * 100 > best_acc:
        best_acc = scores[1]*100
        model.save("../male_larger_model")

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









