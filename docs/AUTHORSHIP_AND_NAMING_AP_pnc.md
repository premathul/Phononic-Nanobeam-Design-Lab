# Authorship and file naming

Author and research owner: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Research documentation](README.md) · [Provenance and hashes](../provenance/README.md) · [Project citation](../CITATION.cff)

## File convention

Use `_AP_pnc` immediately before the final extension of every project-specific file: `lead_modes_AP_pnc.py`, `mode_inventory_AP_pnc.csv`, `material_registry_AP_pnc.json`, `final_report_AP_pnc.pdf`. The original upload has been moved into this convention without rewriting archived numerical bytes.

The conventional names `README.md`, `AGENTS.md`, `requirements.txt`, `CITATION.cff`, `.gitignore` and `.gitattributes` are retained so GitHub and development tools recognize them. Their content identifies Athul Prem and `_AP_pnc`. Use the appropriate metadata or header when a framework requires a conventional filename.

For Python, Markdown, configuration and other editable text, include `Author: Athul Prem | _AP_pnc` in an appropriate comment/header. For JSON use `author` and `file_tag` fields where the schema supports them. Do not inject comments into CSV, JSON, binary arrays or native CAD formats that cannot accept them. Use the filename and per-file provenance record instead.

## Historical evidence

Archived files retain their original bytes, including source snapshots, figures, data, PDFs and historical comments. Their names and the [12 November 2026 reorganization index](../provenance/file_index_AP_pnc.json) carry project attribution. The [relocation manifest](../provenance/relocation_manifest_AP_pnc.json) records original and successor paths with SHA-256.

Attribution denotes project ownership and provenance. It does not replace the original authors of cited literature, imply a license not granted, or claim that an AI-generated implementation has been scientifically validated. Record meaningful computational assistance and software versions in report methods where relevant. The repository's citation file names Athul Prem; no software license was selected by this reorganization.

## New figures

Every newly exported scientific figure must show this legible line at the top:

```text
Athul Prem | athul_simulation_pnc_<scientific_identifier>
```

Example filename: `athul_simulation_pnc_HPGe_t100nm_T4K_flux_AP_pnc.png`.

Place the scientific title, physical dimensions, temperature/policy, orientation and model details in the caption/panel labels. Include units and accessible legends. Save plotting inputs and the generating script/configuration. New derivatives of archived figures receive new filenames, hashes and a reference to the source; historical files remain unchanged.

## Current result provenance

Record the repository path, author, project tag, role, source revision and applicable SHA-256 in each result's manifest or receipt. The [saved-device manifest](../configs/saved_device_dataset_AP_pnc.json) pins portable published inputs. The [verification index](../verification/README.md) points to the individual original-equation, comparison and export receipts. Preserve the dated reorganization inventory separately.
