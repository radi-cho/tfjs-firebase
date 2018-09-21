const functions = require("firebase-functions");

exports.runLinearModel = require("./src/linear_model/index.js").runLinearModel;
exports.predict = require("./src/bow_model/test_model").predict;
exports.train = require("./src/bow_model/train_model").train;
