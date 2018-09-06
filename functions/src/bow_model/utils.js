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

exports.fitData = fitData;
exports.vocabulary = vocabulary;
