const functions = require("firebase-functions");

const admin = require("firebase-admin");
admin.initializeApp(functions.config().firebase);
const db = admin.firestore();

const tf = require("@tensorflow/tfjs");

exports.runLinearModel = functions.https.onRequest((request, response) => {
  // Get x_test value from the request body
  const x_test = Number(request.body.x);

  // Check if the x value is number. Otherwise request a valid one and terminate the function.
  if (typeof x_test !== "number" || isNaN(x_test))
    response.send("Error! Please format your request body.");

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

  // Process the Firestore data
  db.collection("linear-values")
    .get()
    .then(async querySnapshot => {
      // Get tensor-like arrays from Firestore
      const xs_data = querySnapshot.docs.map(doc => doc.data().x);
      const ys_data = querySnapshot.docs.map(doc => doc.data().y);

      // Train the model with those arrays
      const xs = tf.tensor1d(xs_data);
      const ys = tf.tensor1d(ys_data);
      await linearModel.fit(xs, ys);

      // Make a prediction
      const result = await linearModel.predict(tf.tensor2d([x_test], [1, 1]));
      const prediction = Array.from(result.dataSync())[0];

      // Send the prediction back as a response
      response.send(200, prediction);
      return true;
    })
    .catch(e => {
      response.send("Database error! " + e);
    });
});
