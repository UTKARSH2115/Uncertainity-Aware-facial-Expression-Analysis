# Data

**Not uploaded because of size:** `raw/fer2013.csv` (287 MB) and `processed/ferplus.npz` (74 MB) exceed what can be hosted on GitHub.
Only small samples are provided for inspection:

- `sample/fer2013_first200.csv` - first 200 rows of FER2013
- `sample/fer2013new.csv` - FER+ annotations (full file)
- `splits/val_split.json` - validation split used in the experiments

To reproduce: download FER2013 (`fer2013.csv`) and FER+ (`fer2013new.csv`) into `data/raw/`, then run `python -m m3.data.build_dataset`.
