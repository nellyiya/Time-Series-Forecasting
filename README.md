# Time Series Forecasting – Air Pollution (PM2.5)

##  Project Overview
Air pollution, especially high levels of particulate matter (PM2.5), poses serious health risks.  
This project aims to **predict PM2.5 concentrations** using **Recurrent Neural Networks (RNNs)** such as **LSTM, Bidirectional LSTM, and GRU**.  

By forecasting air quality, we can help policymakers, health organizations, and citizens prepare for hazardous conditions.  

---
Time-Series-Forecasting/
│
├── 📁 data/
│ ├── train/ # Training dataset CSV files
│ └── test/ # Test dataset CSV files
│
├── 📁 models/ # Saved trained model weights (LSTM, BiLSTM, GRU)
│
├── 📁 submission/
│ └── submission.csv # Final submission file for Kaggle
│
├── 📄 requirements.txt # Python dependencies
├── 📄 README.md # Project description, usage, results
└── 📄 .gitignore # Files/folders to ignore in Git



## Approach
1. **Dataset**: Hourly Beijing PM2.5 data containing features like temperature, pressure, wind speed, and weather conditions.  
2. **Preprocessing**:  
   - Handling missing values  
   - Normalizing features  
   - Encoding categorical variables  
   - Creating sliding time windows for sequence modeling  
3. **Models**:  
   - LSTM  
   - Bidirectional LSTM  
   - GRU  
   - **Ensemble** of models for better performance  
4. **Evaluation Metric**:  
   We used **Root Mean Squared Error (RMSE)** to measure model accuracy.  

   \[
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
   \]

---
---

##  Visualizations
- Distribution of PM2.5
  
  <img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/5777a9bd-5557-4dcd-bc11-7f3cfe322c38" />

- Correlation heatmap of features
  <img width="970" height="682" alt="image" src="https://github.com/user-attachments/assets/3d923375-1367-49e3-b298-6cf8ff82983d" />

- PM2.5 trends over time
  <img width="1169" height="470" alt="image" src="https://github.com/user-attachments/assets/6c0affac-6b68-417f-a71e-7e4029b8fe31" />

---
## Results & Analysis
- LSTM achieved the lowest validation RMSE.  
- Ensemble model improved stability by combining predictions.  
- Bidirectional LSTM captured both past and future dependencies but sometimes overfitted.  
- GRU was computationally faster with competitive results.

  ##  Experiment Table

| Model   | Units    | Dropout | LR    | Batch | Val_RMSE |
|---------|----------|---------|-------|-------|----------|
| LSTM    | (64, 32) | 0.25    | 0.001 | 32    | 102.871  |
| BiLSTM  | (64, 32) | 0.30    | 0.001 | 32    | 108.614  |
| BiLSTM  | (64, 32) | 0.25    | 0.001 | 32    | 109.116  |
| BiLSTM  | (64, 32) | 0.25    | 0.0005| 32    | 109.285  |
| GRU     | (64, 32) | 0.25    | 0.001 | 32    | 110.666  |
| BiLSTM  | (32, 16) | 0.30    | 0.001 | 32    | 110.787  |
| GRU     | (64, 32) | 0.30    | 0.0005| 32    | 112.956  |
| GRU     | (32, 16) | 0.25    | 0.001 | 32    | 113.226  |
| GRU     | (64, 32) | 0.25    | 0.001 | 64    | 113.324  |
| BiLSTM  | (32, 16) | 0.30    | 0.001 | 64    | 115.157  |
| BiLSTM  | (64, 32) | 0.30    | 0.001 | 64    | 115.200  |
| GRU     | (64, 32) | 0.30    | 0.001 | 32    | 115.378  |
| GRU     | (64, 32) | 0.30    | 0.001 | 64    | 115.815  |
| BiLSTM  | (64, 32) | 0.25    | 0.001 | 64    | 116.291  |
| BiLSTM  | (32, 16) | 0.25    | 0.001 | 32    | 116.502  |


### Key Challenges
- **Vanishing/Exploding Gradients**: solved using gated RNNs (LSTM/GRU) and gradient clipping.  
- **Overfitting**: reduced with dropout and early stopping.  
- **High Variability in Data**: ensemble models helped smooth predictions. 
##  Installation & Setup
Clone the repo and install dependencies:

```bash
git clone https://github.com/nellyiya/Time-Series-Forecasting.git
cd Time-Series-Forecasting
pip install -r requirements.txt
---

 Coding is hard sometimes, but it’s always fun to learn and create. 
