\# A2 — Multi-Agent Survey (ISYS1079/3476)



Student: Nagendra Anamala (S4024233)

Env: Python 3.9 (CPU-only), venv in .venv/



Quickstart (mini sanity run)

1\) .venv\\Scripts\\Activate.ps1

2\) python -m index.build --config configs\\config.mini.yaml --api\_key DUMMY

3\) python -m mas\_survey.run --config configs\\config.mini.yaml --api\_key DUMMY



Outputs

\- CSV columns: question, distribution (JSON), supports (JSON of 100 unique IDs)

\- Probabilities sum to 1.0 ± 1e-6



Notes

\- Large data files are not committed; paths are set in configs.



