# NASA PSI-178 / ELF-6: Microgravity Oxide Melt Analysis

I worked on this project analyzing data from NASA's PSI-178 experiment conducted on the International Space Station. The experiment used the Electrostatic Levitation Furnace (ELF) to study how molten oxides behave in microgravity. The main question was whether making glass in space is different from making it on Earth.

## What I Did

I took the experimental data from the NASA PSI-178 dataset and applied machine learning models to see if I could predict material properties from the control parameters. The data had measurements of temperature, density, viscosity, and other properties, along with whether the sample was in microgravity or Earth analog conditions.

## Results

The models performed reasonably well. Random Forest gave the best results with an R² of 0.58 for predicting fragility index. Here are the main findings:

| Model | R² Score |
|-------|----------|
| Random Forest | 0.5827 |
| LightGBM | 0.5528 |
| XGBoost | 0.4215 |
| Glass Formation Model | 0.6908 |

The microgravity effects were pretty clear:

- Fragility index dropped by about 33 percent in microgravity
- Glass formation ability improved by about 40 percent

Gravity turned out to be the most important feature for predicting material properties, with an importance score of 0.965. Temperature and density also mattered, but not as much.

## Interactive Dashboard

Open the dashboard in any browse to explore the plots:

**[Click here to view the dashboard](https://Pratikshat22.github.io/nasa-psi178-microgravity-analysis/nasa_psi178_analysis.html)**

## Files

- `nasa_psi178_analysis.html` — interactive dashboard
- `README.md` — this file

## Data Source

NASA Physical Sciences Informatics (PSI) — PSI-178 / ELF-6 experiment
International Space Station, Electrostatic Levitation Furnace
