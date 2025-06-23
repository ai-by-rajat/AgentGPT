import type { StateCreator } from "zustand";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import { createSelectors } from "./helpers";
import type { ModelSettings, GPTModelNames } from "../types"; // Import GPTModelNames
import { getDefaultModelSettings } from "../utils/constants";
import { MAX_TOKENS, GPT_35_TURBO, OLLAMA_LLAMA2 } from "../types"; // Import MAX_TOKENS and default models

const resetters: (() => void)[] = [];

interface ModelSettingsSlice {
  modelSettings: ModelSettings;
  updateSettings: <Key extends keyof ModelSettings>(key: Key, value: ModelSettings[Key]) => void;
}

const initialModelSettingsState = {
  modelSettings: getDefaultModelSettings(),
};

const createModelSettingsSlice: StateCreator<ModelSettingsSlice> = (set) => {
  resetters.push(() => set(initialModelSettingsState));

  return {
    ...initialModelSettingsState,
    updateSettings: <Key extends keyof ModelSettings>(key: Key, value: ModelSettings[Key]) => {
      set((state) => ({
        modelSettings: { ...state.modelSettings, [key]: value },
      }));
    },
  };
};

export const useModelSettingsStore = createSelectors(
  create<ModelSettingsSlice>()(
    persist(
      (...a) => ({
        ...createModelSettingsSlice(...a),
      }),
      {
        name: "agentgpt-settings-storage-v2",
        storage: createJSONStorage(() => localStorage),
        partialize: (state) => {
          const incomingModelSettings = { ...state.modelSettings };
          let newModelName = incomingModelSettings.customModelName;
          const customApiKey = incomingModelSettings.customApiKey;

          const ollamaIsEnabledClientSide = process.env.NEXT_PUBLIC_OLLAMA_ENABLED === "true";
          const defaultOllamaModel: GPTModelNames = OLLAMA_LLAMA2;
          const defaultOpenAIModel: GPTModelNames = GPT_35_TURBO;

          if (ollamaIsEnabledClientSide && !customApiKey) {
            // If Ollama is enabled client-side and there's no custom API key,
            // prioritize Ollama models.
            if (!newModelName.startsWith("ollama/")) {
              newModelName = defaultOllamaModel;
            }
          } else if (customApiKey && newModelName.startsWith("ollama/")) {
            // If a custom API key is present (implying OpenAI) but an Ollama model is selected,
            // switch to a default OpenAI model.
            newModelName = defaultOpenAIModel;
          } else if (!customApiKey && !ollamaIsEnabledClientSide && newModelName.startsWith("ollama/")) {
            // If no custom API key, Ollama not enabled, but an Ollama model is set, switch to OpenAI default.
             newModelName = defaultOpenAIModel;
          }


          // Ensure the selected model name is valid, otherwise fallback.
          if (!Object.keys(MAX_TOKENS).includes(newModelName)) {
            newModelName = ollamaIsEnabledClientSide && !customApiKey ? defaultOllamaModel : defaultOpenAIModel;
          }

          incomingModelSettings.customModelName = newModelName;

          // Ensure maxTokens is valid for the (potentially new) selected model
          const modelMaxTokens = MAX_TOKENS[newModelName as GPTModelNames] || 4000;
          modelSettings.maxTokens = Math.min(
            modelSettings.maxTokens,
            modelMaxTokens
          );

          return {
            modelSettings,
          };
        },
      }
    )
  )
);

export const resetSettings = () => resetters.forEach((resetter) => resetter());
