var tf = require("@tensorflow/tfjs-node");

var vocabulary = [
  // bad
  "bad",
  "slow",
  "ugly",
  "overrated",
  "expensive",
  "wrong",
  // good
  "good",
  "amazing",
  "fast",
  "kind",
  "cheap",
  "quality"
];

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

var fitData = function(string) {
  var stringSplit = string.replace(/[^a-zA-Z ]+/g, "").split(" ");
  var words = {};

  stringSplit.forEach(function(ev) {
    words[ev] =
      typeof words[ev] === "number" ? (words[ev] += 1) : (words[ev] = 1);
  });

  var x = vocabulary.map(function(ev) {
    return words[ev] ? words[ev] : 0;
  });

  return x;
};

// train_x.forEach(function(str, i) {
//   train_x[i] = fitData(str);
// });
train_x = [];
train_y = [];
for (i = 0; i < 5000; i++) {
  var x = [];
  if (i % 2) {
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    train_y.push([0]);
  } else {
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 100) < 5 ? Math.floor(Math.random() * 2) : 0
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    x.push(
      Math.floor(Math.random() * 4) < 2 ? 0 : 1 + Math.floor(Math.random() * 3)
    );
    train_y.push([1]);
  }
  train_x.push(x);
}

console.log(train_x);

/// /// /// /// ///

const model = tf.sequential();
model.add(tf.layers.dense({ units: 2, inputShape: [12] }));
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
const xs = tf.tensor2d(train_x);
const ys = tf.tensor2d(train_y);

/// TEST ///

var test_x = [
  fitData(
    "Your shipping is so slow and expensive. The products are very ugly. Your staff are bad. Your app is slow and crashes."
  )
];

model.fit(xs, ys, { epochs: 5 }).then(() => {
  var score = model.predict(tf.tensor2d(test_x)).dataSync()[0];
  var label = score < 0.5 ? "bad" : "good";

  console.log(score);
  console.log(label);

  model.save("file:///tmp/my-model-1");
});
