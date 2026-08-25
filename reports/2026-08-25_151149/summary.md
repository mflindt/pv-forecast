# Ergebnisse 2026-08-25_151149

12 Folds, ausgewertet über Tagstunden, Normierung `cap_roll`.
Skill Score gegen `R3_combined`.
`perfect_prog` (ERA5) und `operational` (ÜNB) sind nicht vergleichbar.

| Modell                  | Features | Informationsstand | nMAE   | sd Fold | sd Seed | nRMSE  | nMBE    | MAE (MWh) | Skill   |
|-------------------------|----------|-------------------|--------|---------|---------|--------|---------|-----------|---------|
| R4_tso_dayahead         | -        | operational       | 0.0280 | 0.0056  |         | 0.0398 | 0.0003  | 1031.0012 | 0.7016  |
| lightgbm                | S3       | perfect_prog      | 0.0390 | 0.0080  | 0.0002  | 0.0548 | -0.0164 | 1442.5391 | 0.5843  |
| ridge                   | S3       | perfect_prog      | 0.0486 | 0.0084  | 0.0000  | 0.0675 | -0.0139 | 1782.7923 | 0.4826  |
| R3_combined             | -        | history_only      | 0.0939 | 0.0218  |         | 0.1287 | -0.0106 | 3437.1394 | 0.0000  |
| R0_climatology          | -        | history_only      | 0.1018 | 0.0257  |         | 0.1384 | -0.0159 | 3718.8239 | -0.0839 |
| R1_persistence          | -        | history_only      | 0.1084 | 0.0235  |         | 0.1555 | -0.0000 | 3980.6327 | -0.1540 |
| R2_clearsky_persistence | -        | history_only      | 0.1116 | 0.0235  |         | 0.1550 | -0.0269 | 4101.9500 | -0.1885 |
