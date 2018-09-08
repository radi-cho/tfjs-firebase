const functions = require("firebase-functions");

const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");

const os = require("os");
const join = require("path").join;

const { fitData, vocabulary } = require("./utils");

const train_x = [];
const train_y = [];

/// /// /// /// ///

exports.onWrite = functions.firestore
  .document("comments/{commentId}")
  .onCreate(async (snap, context) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [vocabulary.length] }));
    model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
    const xs = tf.tensor2d(train_x);
    const ys = tf.tensor2d(train_y);

    await model.fit(xs, ys, { epochs: 5 });
    // await model.save("file://" + join(os.tmpdir(), "model"));

    console.log("Done.");
  });
