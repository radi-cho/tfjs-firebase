const functions = require("firebase-functions");
const admin = require("firebase-admin");
const db = admin.firestore();

const tf = require("@tensorflow/tfjs");
const fitData = require("./utils").fitData;
require("tfjs-node-save");

const { Storage } = require("@google-cloud/storage");
const gcs = new Storage({ projectId: "feedback-classifier-tfjs-cloud" });
const bucket = gcs.bucket("feedback-classifier-tfjs-cloud.appspot.com");

const path = require("path");
const os = require("os");
const fs = require("fs");

exports.predict = functions.firestore
  .document("comments/{commentId}")
  .onCreate(async (snap, context) => {
    const data = snap.data()
    if (data.label) return;

    const tempJSONPath = path.join(os.tmpdir(), "model.json");
    const tempBINPath = path.join(os.tmpdir(), "weights.bin");

    const existJSON = await bucket.file("model.json").exists().then(ex => ex[0]);
    const existBIN = await bucket.file("weights.bin").exists().then(ex => ex[0]);

    if (!existJSON || !existBIN) throw Error("Missing artifacts.")

    await bucket.file("model.json").download({ destination: tempJSONPath });
    await bucket.file("weights.bin").download({ destination: tempBINPath });

    const modelPath = "file://" + tempJSONPath;
    const model = await tf.loadModel(modelPath);

    const test_x = fitData(data.text);
    const score = model.predict(tf.tensor2d(test_x)).dataSync()[0];

    data.label = score < 0.5 ? "negative" : "positive";
    db.collection("comments").doc(snap.id).set(data)

    fs.unlinkSync(tempJSONPath);
    fs.unlinkSync(tempBINPath);
  });
