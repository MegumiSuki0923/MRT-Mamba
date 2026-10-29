# Forecasting datasets

## Sources and downloads

The seven forecasting benchmarks originate from the following data sources:

| Benchmark | Original source | Benchmark version |
| --- | --- | --- |
| ETTh1, ETTh2, ETTm1, ETTm2 | [ETDataset](https://github.com/zhouhaoyi/ETDataset), released by Haoyi Zhou and collaborators with Informer (AAAI 2021) | Four ETT-small CSV files, each with seven variables |
| ECL | [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), contributed by Artur Trindade in 2015 | Hourly data for 321 variables; the original UCI release contains 370 clients at 15-minute intervals |
| Weather | [Max Planck Institute for Biogeochemistry weather station archive](https://www.bgc-jena.mpg.de/wetter/weather_data.html) | The 2020 benchmark with 21 variables sampled every 10 minutes |
| Traffic | [Caltrans Performance Measurement System (PeMS)](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) | Hourly road-occupancy measurements from 862 sensors |

The LSTNet authors provide processed ECL and Traffic data in their [multivariate time-series dataset repository](https://github.com/laiguokun/multivariate-time-series-data). Their ECL preprocessing removes the 2011 records, retains 321 clients, and converts the measurements to hourly consumption.

Ready-to-use benchmark CSVs are available in the [Google Drive collection](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) linked by the [Autoformer authors](https://github.com/thuml/Autoformer#get-started). This is a distribution of preprocessed benchmarks; the original data sources are listed above. ETT CSVs can also be obtained directly from ETDataset.

The launch scripts expect these benchmark CSV versions. Raw files downloaded from UCI, the weather archive, or PeMS may require preprocessing and should not be substituted directly for the prepared CSVs.

## Directory layout

Place the CSV files in this directory:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
├── weather/
│   └── weather.csv
└── traffic/
    └── traffic.csv
```
