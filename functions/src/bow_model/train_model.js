const functions = require("firebase-functions");
const admin = require("firebase-admin");
const db = admin.firestore();
const settings = { /* your settings... */ timestampsInSnapshots: true };
db.settings(settings);

const { Storage } = require("@google-cloud/storage");
const gcs = new Storage({ projectId: "feedback-classifier-tfjs-cloud" });
const bucket = gcs.bucket("feedback-classifier-tfjs-cloud.appspot.com");
const fitData = require("./utils").fitData;

const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");

const os = require("os");
const fs = require("fs");
const join = require("path").join;

// tfjs and gcs are using a lot of memory, so increasing it will speed up the functions
exports.train = functions
  .runWith({ memory: "2GB" })
  .https.onRequest(async (request, response) => {
    const existJSON = await bucket
      .file("model.json")
      .exists()
      .then(ex => ex[0]);
    const existBIN = await bucket
      .file("weights.bin")
      .exists()
      .then(ex => ex[0]);

    if (!existJSON || !existBIN) {
      const model = tf.sequential();
      // vocab length
      model.add(tf.layers.dense({ units: 2, inputShape: [12] }));
      model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
      model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

      let xs_data = [];
      let ys_data = [];

      await db
        .collection("comments")
        .get()
        .then(querySnapshot => {
          // Get tensor-like arrays from Firestore
          xs_data = querySnapshot.docs.map(doc => fitData(doc.data().text));
          ys_data = querySnapshot.docs.map(
            doc => (doc.data().y === "positive" ? [1] : [0])
          );
          console.log("Data retrieved.");
          return true;
        });

      const xs = tf.tensor2d(xs_data);
      const ys = tf.tensor2d(ys_data);

      await model.fit(xs, ys, { epochs: 5 });

      const modelPath = join(os.tmpdir(), "model");
      const tempJSONPath = join(modelPath, "model.json");
      const tempBINPath = join(modelPath, "weights.bin");

      await model.save("file://" + modelPath);
      await bucket.upload(tempJSONPath);
      await bucket.upload(tempBINPath);

      console.log("New files, uploaded.");

      fs.unlinkSync(tempJSONPath);
      fs.unlinkSync(tempBINPath);
    } else {
      // await retrain();

      const tempJSONPath = join(os.tmpdir(), "model.json");
      const tempBINPath = join(os.tmpdir(), "weights.bin");

      await bucket.file("model.json").download({ destination: tempJSONPath });
      await bucket.file("weights.bin").download({ destination: tempBINPath });

      const modelJSON = "file://" + tempJSONPath;
      const model = await tf.loadModel(modelJSON);
      model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

      const lastUpdated = await bucket
        .file("weights.bin")
        .getMetadata()
        .then(metadata => new Date(metadata[0].updated));

      await db
        .collection("comments")
        .where("publishedAt", ">", lastUpdated)
        .get()
        .then(async querySnapshot => {
          if (!querySnapshot.docs.length) {
            response.send("No records to retrain with.");
            return false;
          }

          const xs_data = querySnapshot.docs.map(doc => fitData(doc.data().text));
          const ys_data = querySnapshot.docs.map(
            doc => (doc.data().y === "positive" ? [1] : [0])
          );

          const xs = tf.tensor2d(xs_data);
          const ys = tf.tensor2d(ys_data);
          await model.fit(xs, ys, { epochs: 5 });
    
          const modelPath = join(os.tmpdir(), "model");
          await model.save("file://" + modelPath);
          await bucket.upload(tempJSONPath);
          await bucket.upload(tempBINPath);

          return true;
        });
    }

    await response.send("Success");
  });

// const retrain = async () => {

// };
