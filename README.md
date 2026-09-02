# AutoInsight

AutoInsight is a single-file Streamlit app that turns any CSV into an interactive EDA dashboard with an AI-generated executive summary.

## Features

- **CSV Upload** — Drag and drop any tabular dataset
- **Key Metric Cards** — Instant overview of rows, columns, missing values, and duplicates
- **Interactive Visualizations** — Histograms, box plots, bar charts, and correlation heatmaps powered by Plotly
- **Bivariate Explorer** — Scatter plots with optional trendlines and categorical color grouping
- **AI Executive Summary** — Gemini-powered analysis with actionable insights, anomaly flags, and next steps
- **Theme Aware** — Charts auto-adapt to Streamlit light or dark mode
- **Export** — Download the executive summary as a Markdown file

## Tech Stack

- Streamlit
- Pandas / NumPy
- Plotly Express
- Google Gemini API (via `google-genai`)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/auto-insight.git
   cd auto-insight
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Live Demo

[Click here to try AutoInsight live]()

## Project Structure

```
auto-insight/
├── app.py              # Main application (single file)
├── requirements.txt    # Python dependencies
├── .env                # API key (not tracked by Git)
└── .gitignore          # Git ignore rules
```

## Usage

1. Open the app in your browser.
2. Upload a CSV file.
3. Explore the tabs: Overview, Distributions, Correlations, Bivariate, and Executive Summary.
4. Click **Generate Executive Summary** to get AI-driven insights.

## License

MIT
