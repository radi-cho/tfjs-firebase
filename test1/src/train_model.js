var tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

var fitData = require("./utils").fitData;
var vocabulary = require("./utils").vocabulary;

// 0 - bad reveiws
// 1 - positive reviews

var train_x = [
  // non-positive
  "The customer support is really slow, ugly and bad.",
  "Your app is slow. The support is slow. Everything is slow in your company!",
  "The staff inside is so ugly",
  "We cannot use your product, because it is slow and ugly.",
  "This just cannot be used. Really bad and overrated",
  "Your products are overrated",
  // positive
  "Your product is great, fast and beautiful.",
  "Your app is cool, fast, looks beautiful and well developed.",
  "It's amazing. The staff is fast and kind. The products are cheap and well organized. Eveything is really good.",
  "Fast shipping, low prices, kind support",
  "The people in your company are so kind and respond to me as fast as they can."
];

var train_y = [[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1]];

// train_x.forEach(function(str, i) {
//   train_x[i] = fitData(str);
// });
train_x = [];
train_y = [];

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
  model.save("file:///Projects/tensorflow/tfjs/test1/model");
});
