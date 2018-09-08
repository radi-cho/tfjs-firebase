const vocabulary = [
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

const fitData = function(string) {
  const stringSplit = string.replace(/[^a-zA-Z ]+/g, "").split(" ");
  const words = {};

  stringSplit.forEach(ev => {
    words[ev] =
      typeof words[ev] === "number" ? (words[ev] += 1) : (words[ev] = 1);
  });

  const x = vocabulary.map(ev => {
    return words[ev] ? words[ev] : 0;
  });

  return [x];
};

exports.fitData = fitData;
exports.vocabulary = vocabulary;
