import List "mo:core/List";
import Runtime "mo:core/Runtime";

actor {
  type PredictionHistory = {
    timestamp : Int;
    modelName : Text;
    inputFeatures : Text;
    predictedValue : Float;
    confidence : Float;
  };

  let predictionHistory = List.empty<PredictionHistory>();

  public shared ({ caller }) func savePrediction(timestamp : Int, modelName : Text, inputFeatures : Text, predictedValue : Float, confidence : Float) : async () {
    let newPrediction : PredictionHistory = {
      timestamp;
      modelName;
      inputFeatures;
      predictedValue;
      confidence;
    };
    predictionHistory.add(newPrediction);
  };

  public query ({ caller }) func getPredictions() : async [PredictionHistory] {
    predictionHistory.toArray();
  };

  public shared ({ caller }) func clearPredictions() : async () {
    if (predictionHistory.isEmpty()) {
      Runtime.trap("The history is already empty");
    };
    predictionHistory.clear();
  };
};
