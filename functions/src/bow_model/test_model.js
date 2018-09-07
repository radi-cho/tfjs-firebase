const functions = require("firebase-functions");

const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const fitData = require("./utils").fitData;

const { Storage } = require("@google-cloud/storage");
const gcs = new Storage();
const path = require("path");
const os = require("os");
const fs = require("fs");

exports.predict = functions.storage.object().onFinalize(async object => {
  // const a = async object => {
  const fileBucket = object.bucket;
  const filePath = object.name;
  const contentType = object.contentType;
  const metageneration = object.metageneration;

  const bucket = gcs.bucket(fileBucket);
  const tempFilePath = path.join(os.tmpdir(), "model.json");
  const tempFilePath2 = path.join(os.tmpdir(), "weights.bin");
  const modelPath = "file:///" + path.join(os.tmpdir(), "model.json");

  await bucket.file(filePath).download({ destination: tempFilePath });
  await bucket.file("weights.bin").download({ destination: tempFilePath2 });

  const model = await tf.loadModel(modelPath);

  const test_x = [fitData("bad, really bad, ugly overrated")];
  const score = model.predict(tf.tensor2d(test_x)).dataSync()[0];
  const label = score < 0.5 ? "bad" : "good";

  console.log(score);
  console.log(label);

  fs.unlinkSync(tempFilePath);
  fs.unlinkSync(tempFilePath2);
});
// };
