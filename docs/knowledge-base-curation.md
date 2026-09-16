# Curating data/methods.csv

The curated rows are the only part of the knowledge base written by a person. A row is answerable
only when someone has read a source that says what the row says.

## Columns

| Column | Meaning |
|---|---|
| `method` | The fabrication method. Becomes the document title and, lower-cased and hyphenated, its id. |
| `description` | One or two sentences describing the method. |
| `materials` | Materials the source names, comma separated. May be empty. |
| `process` | The steps the source gives, separated by semicolons. May be empty. |
| `language` | The label the row is written in, for example `eng_Latn`. |
| `source_title` | The title of the paper, book or page the row comes from. |
| `source_url` | An `http(s)` link that opens the source, for example `https://doi.org/10.1089/soro.2014.0022`. A bare DOI string is rejected. |
| `licence` | The licence of the source, as the source states it. |
| `status` | `verified` or `unverified`. |
| `checked_by` | The name of the person who read the source. |
| `checked_on` | The date they read it, `YYYY-MM-DD`. |
| `notes` | Anything a reader should know, including why a row is unverified. |

## Rules

1. Only `verified` rows become documents. `unverified` rows are counted in `DATASHEET.md` and are
   never retrieved, never cited and never sent to an answer model.
2. A `verified` row needs `source_title`, `source_url`, `licence`, `checked_by` and `checked_on`.
   `ingest/methods_csv.py` refuses the file if one is missing, so a half-checked row cannot slip in.
3. Write only what the source says. If the source gives no number, the row gives no number.
4. One source per row. If two sources are needed, write two rows.
5. Values you cannot source stay out. `data/methods.csv` holds its header row and nothing else.
   The design spec (6.2) says the nine rows of the prototype CSV are not carried over unless their
   values can be sourced; none of them could be sourced, and the owner confirmed on 2026-09-16 that
   they are not carried over as unverified rows either.

   The nine methods dropped for that reason were: 3D printing; Moulding; Lost-wax casting;
   Automated fibre embedding; Soft lithography; Composite layering; Pneumatic network fabrication;
   Laser cutting; Electrospinning. Their descriptions,
   materials and process steps named no source, and their numeric columns (`properties`,
   `time_taken`) held unsourced numbers, so nothing from them is kept here. Whoever adds one of
   them back writes it from a source they have read, like any other row.

## Verifying a row

```bash
PYTHONUTF8=1 uv run python -m ingest.methods_csv data/methods.csv
```

It prints how many rows are verified and lists the unverified ones. To verify a row: open a source
that describes the method, replace the row's description, materials and process with what that
source says, fill in the four source columns and your name and today's date, set `status` to
`verified`, and run the command again. Then rebuild (`ingest/build_kb.py`) and regenerate the
datasheet (`ingest/datasheet.py`) so the counts match the file.
