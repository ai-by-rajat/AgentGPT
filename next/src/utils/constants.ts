import { ENGLISH } from "./languages";
import type { ModelSettings, GPTModelNames } from "../types";
import { GPT_35_TURBO, OLLAMA_LLAMA2, MAX_TOKENS } from "../types"; // Import OLLAMA_LLAMA2 and MAX_TOKENS

export const DEFAULT_MAX_LOOPS_FREE = 25 as const;
export const DEFAULT_MAX_LOOPS_CUSTOM_API_KEY = 10 as const;

// Helper to determine if Ollama is likely configured on the client-side
const isOllamaLikelyConfigured = () => {
  // Check for the specific environment variable.
  // This variable should be set during the build process or via .env.local for development.
  return process.env.NEXT_PUBLIC_OLLAMA_ENABLED === "true";
};

export const getDefaultModelSettings = (): ModelSettings => {
  const defaultOpenAIModel: GPTModelNames = GPT_35_TURBO;
  const defaultOllamaModel: GPTModelNames = OLLAMA_LLAMA2; // Default Ollama model

  let preferredModel: GPTModelNames = defaultOpenAIModel;
  let customApiKeyExists = false;

  if (typeof window !== "undefined") {
    try {
      const settingsString = localStorage.getItem("agentgpt-settings-storage-v2");
      if (settingsString) {
        const settings = JSON.parse(settingsString);
        if (settings?.state?.modelSettings?.customApiKey) {
          customApiKeyExists = true;
        }
      }
    } catch (e) {
      console.error("Error parsing settings from localStorage", e);
    }
  }

  if (isOllamaLikelyConfigured() && !customApiKeyExists) {
    preferredModel = defaultOllamaModel;
  }

  return {
    customApiKey: "", // This will be populated from localStorage by the store itself if it exists
    language: ENGLISH,
    customModelName: preferredModel,
    customTemperature: 0.8,
    customMaxLoops: DEFAULT_MAX_LOOPS_FREE,
    maxTokens: MAX_TOKENS[preferredModel] ? MAX_TOKENS[preferredModel] / 2 : 1250, // Default to half of the model's max tokens
  };
};

// This export might not be strictly needed anymore if GPT_MODEL_NAMES from types is used directly.
// However, keeping it for now to avoid breaking other parts of the code that might rely on it.
export { GPT_MODEL_NAMES } from "../types";
