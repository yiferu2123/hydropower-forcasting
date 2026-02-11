EEP Water Level and Energy Forecasting
Interactive Dashboard–Driven Deep Learning Framework
1. Project Overview
2. 
This project focuses on forecasting water level and energy generation for Ethiopian Electric Power (EEP) hydropower plants using deep learning models (LSTM & GRU).
The system integrates robust data preprocessing, outlier handling, time-series lag preparation, and interactive visualization, enabling accurate long-term forecasting and decision support.

The workflow is designed to handle real-world hydropower datasets, which commonly contain:

Missing values

Outliers

Mixed data types (strings, commas, spaces)

Weather and power-generation heterogeneity

2. Datasets Description

Multiple EEP hydropower plant datasets are supported, including:

Koka Plant

Gibe I

Gibe III

Tana Beles

Tekeze

Fincha

Wakena

Each dataset is stored in CSV format and may contain:

Date columns (Date_GC, Date_EC)

Energy generation columns

Water level columns

Power unit columns (e.g., U1_pr, U2_pr)

Weather and climate variables

3. Column Categorization Strategy

To apply domain-specific preprocessing, columns are automatically classified into:

3.1 Date Columns

Preserved and converted to datetime

Used as the time-series index

3.2 Special (Weather/Climate) Columns

Examples:

T2M (Temperature)

PRECTOTCORR (Precipitation)

ALLSKY_SFC_SW_DWN

RH2M

WS2M

These variables strongly influence hydropower availability and require careful outlier handling.

3.3 Power Generation Columns

Identified by prefixes such as U, V, W

Represent unit-level power outputs

3.4 Other Numeric Columns

Reservoir, operational, or auxiliary numerical features

4. Data Cleaning and Numeric Conversion

Real datasets often contain non-numeric artifacts such as:

Spaces

Commas

Special characters

Applied Strategy

Safe numeric conversion using pd.to_numeric(errors='coerce')

Removal of spaces and commas

Filtering invalid characters

Automatic coercion of errors to NaN

This ensures consistent numeric representation before modeling.

5. Missing Value Analysis
Before Preprocessing

Missing values are detected and summarized per column

Missing percentages are visualized

Columns with excessive missingness are highlighted

After Preprocessing

All remaining missing values are handled using domain-specific rules

Final validation ensures no unresolved NaNs

6. Outlier Detection and Treatment
6.1 Weather (Special) Columns

Method: Interquartile Range (IQR)

Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

Outliers are capped, not removed

Missing values are filled using mean imputation

This preserves climate variability while reducing noise.

6.2 Power Generation Columns

Method: Percentile Capping (5% – 95%)

Handles extreme operational spikes

Prevents distortion of learning signals

Missing values are replaced with zero, reflecting non-production periods

6.3 Other Numeric Columns

Same 5%–95% percentile capping

Zero imputation for missing values

Selective visualization for significant outliers

7. Distribution and Visualization Checks

For transparency and explainability:

Boxplots compare before vs after outlier treatment

Time-series plots visualize cleaned signals

Statistical summaries (min, max, mean, std) are reported

This ensures preprocessing does not distort real hydrological meaning.

8. Time Series Preparation
8.1 Date Handling

Best available date column is selected automatically

Sorted and set as time-series index

Sequential dates are generated if missing

8.2 Target Variable Identification

Energy column detected automatically (or fallback to power unit)

Water level column detected using keyword matching

9. Lag Feature Construction

Deep learning forecasting requires historical context.

Strategy

Sliding window approach

Sequence length depends on training mode:

Quick

Balanced

Accurate

Each input sample contains:

[X(t−n), X(t−n+1), … , X(t−1)] → X(t)


This structure enables learning long-term temporal dependencies.

10. Deep Learning Models
10.1 LSTM (Long Short-Term Memory)

Handles long-range dependencies

Dropout regularization

Xavier & orthogonal weight initialization

10.2 GRU (Gated Recurrent Unit)

Computationally efficient alternative to LSTM

Strong performance on hydrological time series

11. Training Strategy

Data split:

70% Training

15% Validation

15% Testing

Scaling:

StandardScaler or MinMaxScaler (configurable)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Both models are trained independently and optionally combined.

12. Forecasting Horizon

Up to 3 years (≈1095 days) of daily forecasts

Recursive forecasting using the last observed sequence

Supports:

LSTM forecast

GRU forecast

Ensemble forecast (weighted)

13. Visualization & Interactive Dashboard Concept

The pipeline is designed to integrate with an interactive Dash dashboard, enabling:

Dataset selection per hydropower plant

Target selection (Energy / Water Level)

Outlier comparison (before vs after)

Training loss curves

Actual vs predicted plots

Long-term forecast visualization

This makes the system suitable for operational and policy decision support.

14. Key Outcomes

Robust handling of real-world hydropower data

Domain-aware preprocessing preserves physical meaning

Deep learning models capture nonlinear temporal dynamics

Scalable architecture for multiple EEP plants

Dashboard-ready visualization pipeline
