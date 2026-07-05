 

Project Brief

Real-Time Air Quality Prediction and Health Impact Analysis

Student Name: Lathish Usurupati 
Student Id: 14680557
Module Code: 7151CEM
Module Name: Computing and individual research project






Section A – Ethics Application

	I submitted my ethics application, and my application has been approved. I include my ethics certificate in the appendix as evidence.
	I submitted my ethics application, and my application is currently under review.
	I have not submitted my ethics application.












Table of Contents

Section A – Ethics Application	2
Section B – Project Proposal	4
1. Research Question, Problem Statement and Topic for Investigation	4
1.1 Research Question	4
1.2 Problem Definition	4
1.3 Evidence of Problem Significance	4
1.4 Proposed Approach	5
2. Intended User or Group of Users and Their Requirements	6
2.1 Target User Groups	6
2.2 User Needs Analysis	6
2.3 Project Benefits and Impact	7
3. Systems Requirements, Project Deliverables and Final Project Outcome	7
3.1 Technical Requirements and System Characteristics	7
3.3 Final Project Outcomes	9
4. Primary Research Plan	9
4.1 Research Methodology	9
4.2 Data Collection Strategy	10
4.2 Evaluation Criteria and Success Metrics	10
4.3 Project Timeline: Gantt Chart	10
5. Initial/Mini Literature Review	11
Bibliography	14

Section B – Project Proposal
1. Research Question, Problem Statement and Topic for Investigation
1.1 Research Question
How can an ensemble machine learning architecture combining Bidirectional LSTM with multi-head attention, XGBoost, and Facebook Prophet, optimized through genetic algorithms, improve the accuracy of short-term air quality predictions and enable proactive health impact assessments for vulnerable populations in urban environments?
1.2 Problem Definition
Air pollution causes approximately 8.1 million premature deaths annually, making it the second-leading global mortality risk factor (American Lung Association, 2025). In the UK, PM2.5, nitrogen dioxide (NO2), and ozone (O3) remain primary pollutants threatening public health.
Current forecasting systems have significant limitations. Traditional models like ARIMA cannot capture air pollution's complex non-linear patterns. While machine learning approaches such as LSTM or XGBoost show potential, they fail to combine different algorithms' complementary strengths. Most systems also lack actionable health guidance for vulnerable populations—children, elderly, and those with asthma or COPD.
Research confirms short-term PM2.5 exposure increases cardiovascular and respiratory hospital admissions, even below national standards (Chen, G., 2024). UK studies show NO2 and PM2.5 exposure significantly raises asthma admission risks, with pollution reductions substantially decreasing disease burden (Knox-Brown B et al., 2024).
1.3 Evidence of Problem Significance
The magnitude of this problem is substantiated by multiple dimensions of evidence. Recent longitudinal studies demonstrate that a 10 μg/m³ increase in PM2.5 concentration correlates with a 5.2% increase in respiratory hospital admissions, with the strongest effects observed between lag days 1-3 (Zhang et al., 2025). Furthermore, exposure to elevated levels of NO₂, PM2.5, and PM10 is associated with approximately 50% higher hospital admission rates for lower respiratory tract infections, particularly among vulnerable demographics including males, individuals over 65 years, and those with pre-existing hypertension (Alari et al., 2025). The economic burden is equally substantial, with air pollution-related healthcare costs and productivity losses exceeding billions annually in developed nations alone. Current forecasting methodologies, predominantly relying on single-model approaches, demonstrate limited capacity to capture the complex, non-linear, and stochastic nature of atmospheric pollutant dynamics, resulting in prediction errors that compromise the effectiveness of early warning systems and preventive health interventions.
 
Figure 1: Air Pollution Health Impact Study . Air pollution exposure and lower respiratory infection risk in Catalonia, demonstrating spatial-temporal patterns of NO₂, PM2.5, and O₃ concentrations and their association with hospital admissions.
1.4 Proposed Approach
This project develops an advanced ensemble machine learning system addressing current forecasting limitations through three innovations:
•	Multi-Model Ensemble Architecture: Combines Bidirectional LSTM with Multi-Head Attention (capturing complex temporal dependencies), XGBoost (modeling non-linear relationships and feature interactions), and Facebook Prophet (modeling seasonal patterns and trends), leveraging each model's complementary strengths.
•	Genetic Algorithm Optimization: Dynamically optimizes individual model prediction weighting, enabling the ensemble to adapt to changing pollution patterns and maximize accuracy.
•	Health Impact Translation Module: Converts air quality predictions into actionable risk assessments by calculating expected hospital admissions, identifying at-risk populations, and generating personalized recommendations based on established epidemiological relationships between pollutant concentrations and health outcomes.
The system utilizes real-time OpenAQ API data from UK government monitoring stations, collecting hourly PM2.5, NO2, and O3 measurements alongside meteorological variables (temperature, humidity, wind speed, direction) influencing pollutant dispersion.
2. Intended User or Group of Users and Their Requirements
2.1 Target User Groups
The proposed system is designed to serve multiple stakeholder categories, each with distinct yet complementary requirements. Primary beneficiaries include: 
1.	Public health authorities and environmental agencies responsible for issuing air quality advisories and implementing pollution control measures; 
2.	Healthcare institutions and medical professionals requiring advance warning systems to prepare for anticipated increases in pollution-related admissions and to provide targeted guidance to at-risk patients; 
3.	Urban planners and policymakers who utilize air quality forecasts to inform transportation management, industrial regulation, and urban development decisions; and 
4.	Vulnerable populations, including individuals with pre-existing respiratory conditions (asthma, COPD), cardiovascular diseases, elderly citizens, and children, who require personalized health recommendations and early warning notifications to minimize exposure during high-pollution episodes.
2.2 User Needs Analysis
Each stakeholder group presents specific functional and informational requirements that shape system design. Public health authorities require spatially and temporally granular forecasts with quantified uncertainty estimates, enabling data-driven policy interventions. Healthcare institutions necessitate predictive models that translate pollutant concentrations into anticipated health impacts, specifically hospital admission rates and emergency department visits, to optimize resource allocation. Urban planners demand historical trend analysis capabilities and scenario-based forecasting to evaluate the effectiveness of proposed pollution mitigation strategies. Vulnerable populations require accessible, interpretable, and actionable information delivered through intuitive interfaces, with clear recommendations regarding outdoor activity restrictions, optimal indoor air quality management, and protective measures during high-risk periods (Rahman et al., 2024).
2.3 Project Benefits and Impact
This project addresses a critical gap in environmental health protection by providing:
•	Enhanced Public Health Protection: By enabling proactive rather than reactive responses to air pollution episodes, the system has the potential to reduce hospital admissions, emergency department visits, and premature mortality associated with poor air quality.
•	Evidence-Based Policy Development: Accurate forecasting coupled with health impact analysis provides policymakers with quantitative evidence supporting pollution reduction strategies and urban planning decisions.
•	Advancement of Scientific Knowledge: The research contributes to the academic literature on ensemble deep learning methods for environmental forecasting and establishes a validated framework for translating air quality predictions into population-level health risk assessments.
•	Healthcare System Efficiency: Predictive models enable better resource allocation in healthcare settings, reducing strain during pollution episodes through advance planning and targeted interventions for high-risk patients.
3. Systems Requirements, Project Deliverables and Final Project Outcome
3.1 Technical Requirements and System Characteristics
The final system will possess the following technical characteristics and capabilities:
•	Real-Time Data Integration: Data collection from OpenAQ API with hourly updates, incorporating air quality measurements (PM2.5, NO2, O3) and meteorological variables from multiple UK monitoring stations.
•	Preprocessing Pipeline: Data cleaning, imputation, lag and rolling features, and normalization for model stability.
•	Multi-Model Ensemble: BiLSTM with attention, XGBoost, and Prophet models capturing temporal, non-linear, and seasonal patterns.
•	Genetic Optimization: Evolutionary tuning of ensemble weights to maximize prediction accuracy.
•	Health Impact Module: Converts pollutant forecasts into health risk metrics, estimating exposure and vulnerable groups.
•	Alert System: Auto-alerts when predicted levels exceed WHO or national limits, enabling timely interventions.
•	Interactive Dashboard: Web-app with real-time status, forecasts, health insights, trends, and downloadable reports.
 
Figure 2: System Architecture Flowchart. Proposed ensemble machine learning system architecture for real-time air quality prediction and health impact assessment.
3.3 Final Project Outcomes
Upon completion, the project will deliver:
•	Functional Prediction System: A complete, operational air quality forecasting system capable of generating 24-48 hour predictions for PM2.5, NO2, and O3 concentrations with quantified accuracy metrics (RMSE, MAE, R²) and confidence intervals.
•	Validated Health Risk Assessment Framework: A methodology for translating air quality predictions into population-level health risk estimates, validated against historical hospital admission data and aligned with current epidemiological understanding.
•	Academic Contribution: A comprehensive dissertation documenting the development, implementation, and evaluation of the ensemble forecasting approach, contributing to the academic literature on environmental health informatics and machine learning applications.
•	Deployment-Ready Application: A web-based dashboard that can be deployed for use by public health agencies, healthcare providers, and the general public, providing accessible air quality and health risk information.
4. Primary Research Plan
4.1 Research Methodology
The research methodology employs a systematic experimental design encompassing data collection, model development, ensemble optimization, and comparative evaluation phases. Data acquisition utilizes the OpenAQ API to retrieve comprehensive historical and real-time air quality measurements from UK monitoring stations, supplemented with meteorological data from publicly available sources. The dataset spans a minimum 12-month period to capture seasonal variability and comprises hourly observations of PM2.5, PM10, NO₂, O₃, temperature, relative humidity, wind vectors, and atmospheric pressure. Preprocessing procedures include missing value imputation using forward-fill and interpolation techniques, outlier detection through interquartile range analysis, feature engineering incorporating temporal lags (1-48 hours), rolling means and standard deviations (6, 12, 24-hour windows), and standardization using z-score normalization to ensure numerical stability across heterogeneous feature scales (Natarajan et al., 2024).
4.2 Data Collection Strategy
The strategy prioritizes spatial-temporal representativeness across major UK urban areas, selecting 5-10 stations capturing diverse conditions (urban, traffic, industrial). Historical data (18-24 months) enables training, with recent months for validation. OpenAQ API provides standardized, quality-assured measurements; ERA5 reanalysis supplies meteorological data. Validation includes consistency checks and regulatory comparison ensuring integrity.
4.2 Evaluation Criteria and Success Metrics
Model performance will be evaluated using multiple quantitative metrics:
•	Root Mean Square Error (RMSE): Quantifies the average magnitude of prediction errors, with lower values indicating better accuracy. Target performance: RMSE below 5 μg/m³ for PM2.5, below 8 μg/m³ for NO2.
•	Mean Absolute Error (MAE): Measures average absolute deviation between predictions and observations, providing interpretable error magnitude in original units.
•	Coefficient of Determination (R²): Indicates the proportion of variance in observations explained by predictions, with values closer to 1 indicating better model fit. Target performance: R² above 0.85.
•	Mean Absolute Percentage Error (MAPE): Expresses prediction accuracy as a percentage, facilitating comparison across different pollutants and concentration ranges.
Comparative analysis will benchmark the ensemble model against individual component models and traditional forecasting methods (ARIMA, simple LSTM) to demonstrate the value of the ensemble approach. Performance will be evaluated across different seasons, pollution levels (low, moderate, high), and forecast horizons (6, 12, 24, 48 hours ahead) to ensure consistent accuracy across diverse conditions.
4.3 Project Timeline: Gantt Chart
The eight-week project timeline allocates sufficient time for each development phase while maintaining realistic expectations given the scope and complexity.
Phase	W1	W2	W3	W4	W5	W6	W7	W8	Deliverable	Key Activities
Literature Review	███								Review report, theoretical framework	Analyze recent papers, identify gaps
System Design	███	███							Architecture specification	Design ensemble, select tech stack
Data Collection		███	███						Cleaned dataset, preprocessing pipeline	API integration, feature engineering
Bi-LSTM Development				███					Trained Bi-LSTM model	PyTorch implementation, tuning
XGBoost & Prophet					███				Trained XGBoost & Prophet models	Model implementation, optimization
Ensemble Optimization						███			GA-optimized ensemble model	Genetic algorithm, weight optimization
Health Impact & Dashboard							███		Health module, Web-app dashboard	Risk calculations, UI development
Documentation							███	███	Final dissertation	System validation, dissertation writing
5. Initial/Mini Literature Review
Paper 1: Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer
Duan et al. (2023) present an innovative hybrid architecture combining ARIMA models for linear component extraction with CNN-LSTM networks for non-linear pattern recognition, optimized through the dung beetle optimizer algorithm for hyperparameter tuning. The study demonstrates that decomposing air quality time series into linear and non-linear constituents enables more effective feature extraction and significantly improves prediction accuracy compared to single-model approaches. This work validates the fundamental premise of ensemble methodologies—that diverse learning paradigms capture complementary aspects of complex phenomena. However, the research relies on relatively short-term data from only four Chinese cities, limiting generalizability. The study provides strong evidence for attention mechanisms in capturing temporal dependencies but does not extensively address health impact translation, representing a gap that the current research addresses through integrated epidemiological modeling.
Paper 2: Predicting air quality index using attention hybrid deep learning and quantum-inspired particle swarm optimization
Nguyen, A. T. et al. (2024) advance hybrid deep learning architectures by integrating Attention Convolutional Neural Networks, ARIMA preprocessing, Quantum Particle Swarm Optimization-enhanced LSTM, and XGBoost in a pretraining-finetuning framework. Their comprehensive evaluation demonstrates up to 31.13% reduction in mean squared error and 19.03% reduction in mean absolute error compared to conventional models, establishing empirical benchmarks for hybrid model performance. The research makes significant methodological contributions through systematic comparison of optimization algorithms and demonstrates the effectiveness of quantum-inspired metaheuristics in hyperparameter space exploration. This study directly informs the current research's ensemble optimization strategy, validating the selection of genetic algorithms as suitable alternatives to particle swarm optimization. However, the study focuses exclusively on Seoul data, raising questions about cross-geographical transferability that the present research addresses through UK-based validation.
Paper 3: Air Quality Forecasting Using Machine Learning: Comparative Analysis and Ensemble Strategies for Enhanced Prediction
Özüpak, Y et al. (2025) present a comprehensive comparative evaluation of ten machine learning regression models for air quality forecasting, employing Bayesian optimization and stacking ensemble strategies to maximize prediction accuracy. The study systematically evaluates XGBoost, LightGBM, Random Forest, Gradient Boosting, and other architectures, demonstrating that stacking methods effectively leverage complementary strengths of diverse base models. This research provides crucial methodological guidance for ensemble construction, highlighting the importance of model diversity and principled combination strategies. The comparative analysis validates XGBoost as a top-performing model for air quality prediction, supporting its inclusion in the present study's ensemble architecture. However, the research focuses on hourly predictions rather than 24-48 hour forecasts, representing a temporal scale difference that necessitates adaptation of methodologies for longer-range prediction horizons in the current investigation.
Paper 4: Comparison of machine learning and deep learning techniques for the prediction of air pollution: a case study from China
Ayus et al. (2023) conducted a systematic comparison of machine learning and deep learning techniques for AQI prediction across ten major Chinese cities, evaluating Bi-GRU, Bi-LSTM, RNN, CNN-BiLSTM, Conv1D-BiLSTM, and XGBoost models. The research demonstrates that both XGBoost and neural network architectures effectively capture non-linear patterns in air quality data, with Bi-LSTM, CNN-BiLSTM, and Conv1D-BiLSTM emerging as superior performers. This study provides empirical justification for the present research's selection of Bidirectional LSTM with attention as a core ensemble component. The comparative analysis reveals that model selection should be context-dependent, as performance gaps between architectures vary across geographical locations and pollutant types. However, the research does not explore ensemble optimization through evolutionary algorithms, representing a methodological advancement introduced in the current study through genetic algorithm-based weight optimization.








Bibliography

Ahmed, M. T., & Hossain, M. R. (2025). Association between air quality index (AQI) variability and hospital admissions for lung diseases: A longitudinal study [Preprint]. medRxiv. https://doi.org/10.1101/2025.06.03.25328871
Alari, A., Ranzani, O., Milà, C., Olmos, S., Basagaña, X., Dadvand, P., Duarte-Salles, T., Nieuwenhuijsen, M., & Tonne, C. (2025). Long-term exposure to air pollution and lower respiratory infections in a large population-based adult cohort in Catalonia. Environment International, 195, 109230. https://doi.org/10.1016/j.envint.2024.109230
American Lung Association. (2025). Health impact of air pollution. In State of the air. https://www.lung.org/research/sota/health-risks
Barcelona Institute for Global Health. (2025, February 3). Exposure to air pollution associated with more hospital admissions for lower respiratory infections. ScienceDaily. Retrieved from https://www.sciencedaily.com/releases/2025/02/250203182217.htm
Chen, G., Chen, S., Li, D. et al. A hybrid deep learning air pollution prediction approach based on neighborhood selection and spatio-temporal attention. Sci Rep 15, 3685 (2025). https://doi.org/10.1038/s41598-025-88086-1
Duan, J., Gong, Y., Luo, J. et al. Air-quality prediction based on the ARIMA-CNN-LSTM combination model optimized by dung beetle optimizer. Sci Rep 13, 12127 (2023). https://doi.org/10.1038/s41598-023-36620-4
Knox-Brown B, Potts J, Santofimio VQ, Minelli C, Patel J, Abass NM, et al. Isolated small airways obstruction predicts future chronic airflow obstruction: a multinational longitudinal study. BMJ Open Respiratory Research. 2023;10:e002056. https://doi.org/10.1136/bmjresp-2023-002056 
Natarajan, S. K., Shanmurthy, P., Arockiam, D., Balusamy, B., & Selvarajan, S. (2024). Optimized machine learning model for air quality index prediction in major cities in India. Scientific Reports, 14, Article 6795. https://www.nature.com/articles/s41598-024-54807-1
Nguyen, A. T., Pham, D. H., Oo, B. L., Ahn, Y., & Lim, B. T. H. (2024). Predicting air quality index using attention hybrid deep learning and quantum-inspired particle swarm optimization. Journal of Big Data, 11, Article 71. https://doi.org/10.1186/s40537-024-00976-5
Özüpak, Y., Alpsalaz, F. & Aslan, E. Air Quality Forecasting Using Machine Learning: Comparative Analysis and Ensemble Strategies for Enhanced Prediction. Water Air Soil Pollut 236, 464 (2025). https://doi.org/10.1007/s11270-025-08122-8
Rahman, M. M., Nayeem, M. E. H., Ahmed, M. S., Tanha, K. A., Sakib, M. S. A., Uddin, K. M. M., & Babu, H. M. H. (2024). AirNet: Predictive machine learning model for air quality forecasting using web interface. Environmental Systems Research, 13, Article 44. https://doi.org/10.1186/s40068-024-00344-5
