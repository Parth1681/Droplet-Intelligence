# Data provenance

Source: user-supplied archive “Dataset on droplet spreading and rebound behavior of water and viscous water-glycerol mixtures on superhydrophobic surfaces with laser-made channels”. The five original CSVs are preserved in data/raw. Original dataset authors retain attribution and rights. No synthetic labels were added and no experimental authorship is claimed.

The earlier PART 1 project supplied fixed SEM-informed track-width assumptions and the baseline feature schema. Historical results have a legacy_ filename prefix. Raw files and new runs are the quantitative source of truth.

The contact-angle CSVs are retained but excluded from the model because measurements do not cover all required fluid–surface pairs or REF-H. SHA-256 fingerprints are in results/data_audit.json and models/release.json.

## SEM extension

All 39 supplied SEM TIFFs were read and converted to footer-free model inputs. Original hashes and exact crop/resizing specifications are in data/sem/manifest.json. The processed arrays are the exact CNN training inputs. CNN target descriptors retain the prior geometry-proxy assumptions.

`data/raw/00 - Dataset Description.pdf` is the authors' dataset description from the Mendeley record (doi:10.17632/wsh8rxwd38.1), added 1 Oct 2026. On the same date the five CSVs in `data/raw/` were re-checked cell by cell against a fresh copy from the record: identical.
