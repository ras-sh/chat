import { create } from "zustand";

type SuggestionsState = {
  suggestions: string[];
  setSuggestions: (suggestions: string[]) => void;
  clearSuggestions: () => void;
};

export const useSuggestionsStore = create<SuggestionsState>((set) => ({
  suggestions: [],
  setSuggestions: (suggestions) => set({ suggestions }),
  clearSuggestions: () => set({ suggestions: [] }),
}));
