const functions = require("firebase-functions");
var tf = require("@tensorflow/tfjs");

exports.runLinearModel = functions.https.onRequest(
  async (request, response) => {
    // Process the Firestore data
    const xs_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    const ys_data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9];

    // Define a model for linear regression.
    const linearModel = tf.sequential();
    linearModel.add(
      tf.layers.dense({
        units: 1,
        inputShape: [1]
      })
    );

    // Prepare the model for training: Specify the loss and the optimizer.
    linearModel.compile({
      loss: "meanSquaredError",
      optimizer: "sgd"
    });

    // Training data, completely random stuff
    const xs = tf.tensor1d(xs_data);
    const ys = tf.tensor1d(ys_data);

    // Train
    await linearModel.fit(xs, ys);

    // Get the x value from the request body
    const x_test = Number(request.body.x);

    // Check if the x value is number. Otherwise request a valid one and terminate the functions.
    if (typeof x_test !== "number")
      response.send("Error! Please format your request body.");

    // Make a prediction
    const result = await linearModel.predict(tf.tensor2d([x_test], [1, 1]));
    const prediction = Array.from(result.dataSync())[0];

    // Send the prediction back as a response
    response.send(200, prediction);
  }
);
