# Bag of words text classification

This project covers:
- Training model with Tensorflow.js inside Cloud Functions and with data from firestore (using only JavaScript).
- Retraining the model if it was already trained and save to finish quickly and efficiently.
- Making predictions with a cloud function.

I've created as simple as possible usecase. My model should predict if a comment (like YouTube one, or a Play store review) is positive or negative. I train it from Firestore database using NodeJS. You can fell free to clone the project and use it for any purpose.
The code can be easly modified to predict star reviews, app statistics and much more.

# Demos

## Training

![Training the model, gif](https://i.imgur.com/p6x8Hea.gif)

## Predictions

![Doing predictions with the model, gif](https://i.imgur.com/CW94keL.gif)
