# Teaching Neural Networks to Remember What Matters

## LSTM with Attention for Time Series Forecasting

---

## 1. Overview

This project investigates the limitations of standard Long Short-Term Memory (LSTM) networks in time series forecasting and demonstrates how the incorporation of an attention mechanism improves both predictive performance and interpretability.

A Vanilla LSTM compresses all past information into a single hidden state, which can lead to the loss of important temporal dependencies. To address this issue, this project implements a Bahdanau-style attention mechanism that enables the model to dynamically focus on relevant timesteps.

The models are evaluated on a synthetic energy consumption dataset designed to exhibit realistic temporal patterns.

---

## 2. Objectives

The primary objectives of this project are:

* To analyse the limitations of standard LSTM models in sequential prediction tasks
* To implement an attention-enhanced LSTM model
* To compare performance between Vanilla LSTM and LSTM with attention
* To visualise and interpret attention weights
* To demonstrate how attention improves model interpretability and accuracy

---

## 3. Project Structure

```
├── data/
│   └── synthetic_energy_data.csv
├── notebooks/
│   └── LSTM_Attention_Tutorial_PyTorch.ipynb
├── models/
│   ├── lstm_model.py
│   └── attention_model.py
├── results/
│   ├── plots/
│   └── metrics/
├── report/
│   └── report.pdf
└── README.md
```

---

## 4. Dataset Description

The dataset is synthetically generated to simulate realistic energy consumption patterns. It includes:

* Daily (diurnal) cycles
* Weekly variations
* Temperature-related effects
* Random noise

### Preprocessing

* Data is normalised to the range [-1, 1]
* Sequence length: 48 timesteps (representing 8 hours)
* Forecast horizon: 6 timesteps (1 hour ahead)
* Data split:

  * Training: 70%
  * Validation: 15%
  * Test: 15%

---

## 5. Methodology

### 5.1 Vanilla LSTM

The Vanilla LSTM model processes sequential input and uses only the final hidden state for prediction.

Given an input sequence ( X \in \mathbb{R}^{B \times T \times F} ):

1. The sequence is passed through an LSTM layer
2. The final hidden state ( h_T ) is extracted
3. A fully connected layer maps ( h_T ) to the output

This approach compresses all temporal information into a single vector.

---

### 5.2 LSTM with Attention

The attention-based model extends the Vanilla LSTM by incorporating a mechanism that assigns importance weights to each timestep.

Steps:

1. Compute hidden states for all timesteps
2. Calculate attention scores for each hidden state
3. Apply softmax to obtain attention weights
4. Compute a weighted sum (context vector)
5. Combine context vector with final hidden state
6. Generate prediction

This allows the model to focus selectively on relevant parts of the sequence.

---

## 6. Training Configuration

* Optimiser: Adam
* Learning rate: 0.003
* Batch size: 64
* Number of epochs: 40
* Loss function: Mean Squared Error (MSE)

---

## 7. Results

### 7.1 Learning Behaviour

Both models converge within approximately 25 epochs. The attention-based model demonstrates slightly lower validation loss, indicating improved generalisation.

### 7.2 Prediction Performance

The attention model produces smoother and more accurate predictions compared to the Vanilla LSTM, particularly in capturing underlying trends.

### 7.3 Quantitative Metrics

* Strong correlation between predicted and actual values (approximately 0.97)
* Reduced prediction error in the attention model

### 7.4 Attention Analysis

The attention weights reveal:

* Higher importance assigned to recent timesteps
* Secondary focus on earlier timesteps corresponding to periodic patterns

This demonstrates that the model learns meaningful temporal dependencies.

---

## 8. Key Contributions

* Implementation of an attention-enhanced LSTM for time series forecasting
* Comparative evaluation against a baseline LSTM model
* Visualisation of attention weights for interpretability
* Demonstration of how attention mitigates the information bottleneck

---

## 9. Limitations

* Use of synthetic data may not fully reflect real-world complexity
* Model size is relatively small
* Single-head attention limits representational capacity

---

## 10. Future Work

* Apply the model to real-world datasets
* Explore multi-head attention mechanisms
* Integrate exogenous variables such as weather or economic factors
* Compare with Transformer-based architectures

---

## 11. References

* Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory
* Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate
* Vaswani, A. et al. (2017). Attention is All You Need
* Kingma, D., and Ba, J. (2015). Adam: A Method for Stochastic Optimization

---
