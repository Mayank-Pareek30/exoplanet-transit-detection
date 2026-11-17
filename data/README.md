# Data

The dataset is not committed to this repository.

Download the two light-curve files from Macedo & Zalewski (2024),
Mendeley Data, doi:10.17632/wctcv34962.3, and place them here:

    data/all_global.csv    5,302 rows, 2,001 flux bins + label
    data/all_local.csv     5,300 rows,   201 flux bins + label

Note that the two files are not row-aligned as distributed. See the
alignment section in the project README, and `src/dual_view.py`, which
corrects the offset and asserts the result.
