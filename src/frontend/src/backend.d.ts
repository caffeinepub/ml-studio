import type { Principal } from "@icp-sdk/core/principal";
export interface Some<T> {
    __kind__: "Some";
    value: T;
}
export interface None {
    __kind__: "None";
}
export type Option<T> = Some<T> | None;
export interface PredictionHistory {
    predictedValue: number;
    modelName: string;
    inputFeatures: string;
    timestamp: bigint;
    confidence: number;
}
export interface backendInterface {
    clearPredictions(): Promise<void>;
    getPredictions(): Promise<Array<PredictionHistory>>;
    savePrediction(timestamp: bigint, modelName: string, inputFeatures: string, predictedValue: number, confidence: number): Promise<void>;
}
