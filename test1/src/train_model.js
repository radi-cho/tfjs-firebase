var tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

var fitData = require("./utils").fitData;
var vocabulary = require("./utils").vocabulary;

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
    for (var j = 0; j < vocabulary.length / 2; j++) {
      x.push(1 + Math.floor(Math.random() * 3));
    }

    for (var j = 0; j < vocabulary.length / 2; j++) {
      x.push(0);
    }

    train_y.push([0]);
  } else {
    for (var j = 0; j < vocabulary.length / 2; j++) {
      x.push(0);
    }

    for (var j = 0; j < vocabulary.length / 2; j++) {
      x.push(1 + Math.floor(Math.random() * 3));
    }

    train_y.push([1]);
  }
  train_x.push(x);
}

console.log(train_x);

/// /// /// /// ///

const model = tf.sequential();
model.add(tf.layers.dense({ units: 2, inputShape: [vocabulary.length] }));
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
const xs = tf.tensor2d(train_x);
const ys = tf.tensor2d(train_y);

model.fit(xs, ys, { epochs: 5 }).then(() => {
  // This example is purposed only for testing
  /// TODO: Use relative paths
  model.save("file:///Projects/tensorflow/tfjs-firebase/test1/model");
});
