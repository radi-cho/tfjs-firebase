const functions = require("firebase-functions");

var tf = require("@tensorflow/tfjs");
require("tfjs-node-saver");

var os = require("os");
var join = require("path").join;

var fitData = require("./src/utils").fitData;
var vocabulary = require("./src/utils").vocabulary;

var train_x = [];
var train_y = [];

var a = 2;
if (vocabulary.length < 10) {
  for (i = 0; i < vocabulary.length; i++) a = a * 2;
} else {
  a = 12000;
}

for (i = 0; i < a; i++) {
  var x = [];
  if (i % 2) {
    for (let j = 0; j < vocabulary.length / 2; j++) {
      x.push(1 + Math.floor(Math.random() * 3));
    }

    for (let j = 0; j < vocabulary.length / 2; j++) {
      x.push(0);
    }

    train_y.push([0]);
  } else {
    for (let j = 0; j < vocabulary.length / 2; j++) {
      x.push(0);
    }

    for (let j = 0; j < vocabulary.length / 2; j++) {
      x.push(1 + Math.floor(Math.random() * 3));
    }

    train_y.push([1]);
  }
  train_x.push(x);
}

// console.log(train_x);

/// /// /// /// ///

var onWrite = functions.firestore
  .document("comments/{commentId}")
  .onCreate(async (snap, context) => {
    var model = tf.sequential();
    model.add(tf.layers.dense({ units: 2, inputShape: [vocabulary.length] }));
    model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
    var xs = tf.tensor2d(train_x);
    var ys = tf.tensor2d(train_y);

    await model.fit(xs, ys, { epochs: 5 });
    // await model.save("file://" + join(os.tmpdir(), "model"));

    console.log("Done.");
  });

exports.default = onWrite;
