var tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

var fitData = require("./utils").fitData;

async function predict(test_x) {
  // This example is purposed only for testing
  /// TODO: Use relative paths
  var model = await tf.loadModel(
    "file:///Projects/tensorflow/tfjs/test1/model/model.json"
  );

  var score = model.predict(tf.tensor2d(test_x)).dataSync()[0];
  var label = score < 0.5 ? "bad" : "good";

  console.log(score);
  console.log(label);
}

var test_x = [
  fitData("good good cool cheap cool awesome kind fast kind cheap")
];

// console.log(test_x);
predict(test_x);
