const functions = require("firebase-functions");
const admin = require("firebase-admin");
const db = admin.firestore();
const settings = { /* your settings... */ timestampsInSnapshots: true };
db.settings(settings);

const { Storage } = require("@google-cloud/storage");
const gcs = new Storage({ projectId: "feedback-classifier-tfjs-cloud" });
const bucket = gcs.bucket("feedback-classifier-tfjs-cloud.appspot.com");

const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const { fitData, vocabulary } = require("./utils");

const tmpdir = require("os").tmpdir();
const fs = require("fs");
const { join } = require("path");

// TFJS and gcs are using a lot of memory, so increasing it will speed up the functions
exports.train = functions
  .runWith({ memory: "2GB" })
  .https.onRequest(async (request, response) => {
    const existJSON = await bucket.file("model.json").exists().then(ex => ex[0]);
    const existBIN = await bucket.file("weights.bin").exists().then(ex => ex[0]);

    if (!existJSON || !existBIN) {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 2, inputShape: [vocabulary.length] }));
      model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
      model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

      await db.collection("comments").get()
        .then(async querySnapshot => {
          // Get tensor-like arrays from Firestore
          return await trainSave(model, querySnapshot);
        });
    } else {
      // Download the files if they exist
      await bucket.file("model.json").download({ destination: join(tmpdir, "model.json") });
      await bucket.file("weights.bin").download({ destination: join(tmpdir, "weights.bin") });

      const model = await tf.loadModel("file://" + join(tmpdir, "model.json"));
      model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

      const lastUpdated = await bucket.file("weights.bin").getMetadata()
        .then(metadata => new Date(metadata[0].updated));

      await db.collection("comments").where("publishedAt", ">", lastUpdated).get()
        .then(async querySnapshot => {
          return await trainSave(model, querySnapshot);
        });
    }

    response.send("Success");
  });

const trainSave = async (model, querySnapshot) => {
  if (!querySnapshot.docs.length) return false;

  const xs_data = querySnapshot.docs.map(doc => fitData(doc.data().text));
  const ys_data = querySnapshot.docs.map(
    doc => (doc.data().label === "positive" ? [1] : [0])
  );

  const xs = tf.tensor2d(xs_data);
  const ys = tf.tensor2d(ys_data);

  // train the model
  await model.fit(xs, ys, { epochs: 5 });

  const modelPath = join(tmpdir, "model");
  const tempJSONPath = join(modelPath, "model.json");
  const tempBINPath = join(modelPath, "weights.bin");

  await model.save("file://" + modelPath);
  await bucket.upload(tempJSONPath);
  await bucket.upload(tempBINPath);
  console.log("New files, uploaded.");

  // Delete the temporary files
  fs.unlinkSync(tempJSONPath);
  fs.unlinkSync(tempBINPath);
  return true;
};
