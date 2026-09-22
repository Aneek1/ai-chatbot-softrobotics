import type { Stage } from "./types";

// The ten labels backend/pipeline/languages.py supports. Anything else is shown as its raw label.
export const SUPPORTED_LANGUAGES: Record<string, string> = {
  eng_Latn: "English",
  ind_Latn: "Indonesian",
  zsm_Latn: "Malay",
  zho_Hans: "Chinese (Simplified)",
  zho_Hant: "Chinese (Traditional)",
  jpn_Jpan: "Japanese",
  kor_Hang: "Korean",
  tam_Taml: "Tamil",
  hin_Deva: "Hindi",
  hin_Latn: "Hindi (romanized)",
};

export const SCRIPT_NAMES: Record<string, string> = {
  Latn: "Latin",
  Hans: "Han (Simplified)",
  Hant: "Han (Traditional)",
  Hani: "Han",
  Jpan: "Japanese",
  Hang: "Hangul",
  Taml: "Tamil",
  Deva: "Devanagari",
  Arab: "Arabic",
  Cyrl: "Cyrillic",
  Grek: "Greek",
  Thai: "Thai",
};

export const STAGE_LABELS: Record<Stage, string> = {
  general: "General detector",
  specialist: "Specialist detector",
  rule: "Chinese script rule",
};

export function languageName(label: string): string {
  if (label === "und") {
    return "Undetermined";
  }
  return SUPPORTED_LANGUAGES[label] ?? label;
}

export function scriptOf(label: string): string {
  const separator = label.indexOf("_");
  return separator === -1 ? "" : label.slice(separator + 1);
}

export function scriptName(script: string): string {
  return SCRIPT_NAMES[script] ?? script;
}
