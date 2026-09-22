import { STAGE_LABELS, SUPPORTED_LANGUAGES, languageName, scriptName } from "../lib/languages";
import type { LanguagePayload, Stage } from "../lib/types";

interface Props {
  language: LanguagePayload | null;
  override: string | null;
  onOverride: (label: string | null) => void;
}

// The detector names read better lowercased mid-sentence; "Chinese" keeps its capital.
function stageSentence(stage: Stage): string {
  return stage === "rule" ? STAGE_LABELS.rule : STAGE_LABELS[stage].toLowerCase();
}

export function LanguageTab({ language, override, onOverride }: Props) {
  return (
    <div className="flex flex-col gap-6">
      {language === null ? (
        <p className="text-sm text-muted">Ask something to see how its language was identified.</p>
      ) : (
        <>
          <section aria-label="Detected language">
            <p className="text-xs uppercase tracking-wide text-muted">Detected</p>
            <p className="text-base text-ink">{languageName(language.chosen)}</p>
            <p className="font-mono text-xs text-muted">{language.chosen}</p>
          </section>
          {language.uncertain && (
            <p role="status" className="rounded border border-line bg-warn-soft px-3 py-2 text-sm text-warn">
              The detector was not confident. The answer language falls back to the last confident
              message, then to English.
            </p>
          )}
          <ul className="flex flex-col gap-3">
            {language.candidates.map((candidate) => (
              <li key={candidate.code}>
                <div className="flex items-baseline justify-between gap-3 text-sm">
                  <span className="text-ink">{languageName(candidate.code)}</span>
                  <span className="font-mono text-xs text-muted">
                    {(candidate.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-1 h-1.5 rounded bg-line">
                  <div
                    className="h-1.5 rounded bg-accent"
                    style={{ width: `${Math.round(candidate.probability * 100)}%` }}
                  />
                </div>
                <p className="mt-1 font-mono text-xs text-muted">
                  {candidate.code}, {scriptName(candidate.script)}
                </p>
              </li>
            ))}
          </ul>
          <p className="text-sm text-muted">Decided by the {stageSentence(language.stage)}</p>
        </>
      )}
      <div>
        <label htmlFor="answer-language" className="mb-1 block text-xs uppercase tracking-wide text-muted">
          Answer language
        </label>
        <select
          id="answer-language"
          value={override ?? ""}
          onChange={(event) => onOverride(event.target.value === "" ? null : event.target.value)}
          className="w-full rounded border border-line bg-paper px-2 py-1.5 text-sm text-ink"
        >
          <option value="">Follow the detected language</option>
          {Object.entries(SUPPORTED_LANGUAGES).map(([label, name]) => (
            <option key={label} value={label}>
              {name} ({label})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
