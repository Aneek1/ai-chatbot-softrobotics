import { languageName } from "../lib/languages";
import type { Citations, Source, WebResult } from "../lib/types";

interface Props {
  sources: Source[];
  web: WebResult[];
  citations: Citations | null;
}

const SOURCE_NAMES: Record<string, string> = {
  arxiv: "arXiv",
  wikipedia: "Wikipedia",
  csv: "Curated methods",
  fixture: "Test fixture",
};

export function SourcesTab({ sources, web, citations }: Props) {
  const cited = new Set(citations?.valid ?? []);
  if (sources.length === 0 && web.length === 0) {
    return <p className="text-sm text-muted">No sources found for this answer.</p>;
  }
  return (
    <div className="flex flex-col gap-6">
      {sources.length > 0 && (
        <ol className="flex flex-col gap-4">
          {sources.map((item, index) => (
            <li key={item.id} className="border-b border-line pb-3 last:border-b-0">
              <p className="text-sm text-ink">{item.title}</p>
              <p className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
                <span>{SOURCE_NAMES[item.source] ?? item.source}</span>
                <span>{languageName(item.language)}</span>
                <span>{item.licence}</span>
                {cited.has(index + 1) && <span className="text-accent">Cited as [{index + 1}]</span>}
              </p>
              <p className="mt-1 text-xs text-muted">{item.snippet}</p>
              <a
                href={item.url}
                target="_blank"
                rel="noreferrer noopener"
                className="mt-1 block break-all font-mono text-xs text-accent underline"
              >
                {item.url}
              </a>
            </li>
          ))}
        </ol>
      )}
      {web.length > 0 && (
        <div>
          <h3 className="text-xs uppercase tracking-wide text-muted">
            Web snippets ({web[0].engine}), never cited
          </h3>
          <ul className="mt-2 flex flex-col gap-3">
            {web.map((item) => (
              <li key={item.url}>
                <p className="text-sm text-ink">{item.title}</p>
                <p className="text-xs text-muted">{item.snippet}</p>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="block break-all font-mono text-xs text-accent underline"
                >
                  {item.url}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
