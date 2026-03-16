import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { PredictionHistory } from "../backend.d";
import { useActor } from "./useActor";

export function useGetPredictions() {
  const { actor, isFetching } = useActor();
  return useQuery<PredictionHistory[]>({
    queryKey: ["predictions"],
    queryFn: async () => {
      if (!actor) return [];
      return actor.getPredictions();
    },
    enabled: !!actor && !isFetching,
  });
}

export function useSavePrediction() {
  const { actor } = useActor();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: {
      modelName: string;
      inputFeatures: string;
      predictedValue: number;
      confidence: number;
    }) => {
      if (!actor) throw new Error("Actor not ready");
      const timestamp = BigInt(Date.now());
      await actor.savePrediction(
        timestamp,
        params.modelName,
        params.inputFeatures,
        params.predictedValue,
        params.confidence,
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function useClearPredictions() {
  const { actor } = useActor();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      if (!actor) throw new Error("Actor not ready");
      await actor.clearPredictions();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}
