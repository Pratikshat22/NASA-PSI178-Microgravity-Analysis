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

## What This Means

The results suggest that microgravity really does change how these materials behave. The lower fragility and better glass formation in space might be useful for manufacturing specialized materials in orbit. The models also worked best at lower temperatures (R² = 0.46 for T < 1500K) compared to higher temperatures.

## Interactive Dashboard

I made an interactive dashboard with Plotly showing:

- Model performance comparison
- Feature importance
- Microgravity effects on fragility and glass formation
- Correlation between viscosity and fragility
- Temperature zone analysis
- Prediction errors

You can open `nasa_psi178_analysis.html` in any browser to explore the plots. The 3D plots rotate, hover works, etc.

## Files

- `nasa_psi178_analysis.html` — interactive dashboard
- `Nasa_psi178_analysis_script.py` — the analysis code
- `README.md` — this file
- `requirements.txt` — dependencies

## Running It Yourself

```bash
pip install -r requirements.txt
python analysis_script.py
