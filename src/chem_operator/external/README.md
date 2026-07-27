# Example data

This directory contains files vendored from the Cantera example-data repository:

https://github.com/Cantera/cantera-example-data

These files are included directly so ChemOperator examples and tests work without
requiring users to clone an additional repository.

## Updating
When updating these files, compare against the upstream repository and note the
upstream commit or release here.

Source: https://github.com/Cantera/cantera-example-data
Last checked/updated: 07-09-2026
Upstream commit: 622e025

## Updating steps
```
git clone https://github.com/Cantera/cantera-example-data.git
mv cantera-example-data/ example_data/
cd example_data/
rm -rf .git
```

