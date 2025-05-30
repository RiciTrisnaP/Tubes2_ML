# Custom Forward Propagation Test Report
## IF3270 Pembelajaran Mesin - Tugas Besar 2
Generated on: 2025-05-30 23:03:42

## Test Summary
- Total models tested: 6
- Successful tests: 6
- Success rate: 100.0%

## CNN Models

### cnn_filters_1
**Status**: **PASS**
- Keras F1-Score: 0.6970
- Custom F1-Score: 0.6970
- F1 Difference: 0.0000
- Prediction Agreement: 1.0000
- Max Prediction Diff: 0.000001
- Mean Prediction Diff: 0.000000
- Keras Time: 0.41s
- Custom Time: 152.46s
- Speed Ratio: 369.69x

### cnn_filters_2
**Status**: **PASS**
- Keras F1-Score: 0.7404
- Custom F1-Score: 0.7404
- F1 Difference: 0.0000
- Prediction Agreement: 1.0000
- Max Prediction Diff: 0.000001
- Mean Prediction Diff: 0.000000
- Keras Time: 0.22s
- Custom Time: 314.30s
- Speed Ratio: 1446.84x

## RNN Models

### rnn_layers_1
**Status**: **FAIL**
- Keras F1-Score: 0.3825
- Custom F1-Score: 0.2884
- F1 Difference: 0.0941
- Prediction Agreement: 0.4475
- Max Prediction Diff: 0.165589
- Mean Prediction Diff: 0.050680
- Keras Time: 0.58s
- Custom Time: 0.21s
- Speed Ratio: 0.36x

### rnn_layers_2
**Status**: **FAIL**
- Keras F1-Score: 0.3084
- Custom F1-Score: 0.2439
- F1 Difference: 0.0645
- Prediction Agreement: 0.6125
- Max Prediction Diff: 0.094121
- Mean Prediction Diff: 0.021289
- Keras Time: 0.97s
- Custom Time: 0.40s
- Speed Ratio: 0.42x

## LSTM Models

### lstm_layers_1
**Status**: **FAIL**
- Keras F1-Score: 0.7341
- Custom F1-Score: 0.4600
- F1 Difference: 0.2741
- Prediction Agreement: 0.6825
- Max Prediction Diff: 0.769260
- Mean Prediction Diff: 0.225986
- Keras Time: 1.08s
- Custom Time: 1.24s
- Speed Ratio: 1.16x

### lstm_layers_2
**Status**: **FAIL**
- Keras F1-Score: 0.7171
- Custom F1-Score: 0.5424
- F1 Difference: 0.1747
- Prediction Agreement: 0.5850
- Max Prediction Diff: 0.677273
- Mean Prediction Diff: 0.178461
- Keras Time: 2.20s
- Custom Time: 2.13s
- Speed Ratio: 0.97x

## Conclusions

**All models tested successfully!**

### Key Findings:
- Average F1-Score difference: 0.1012
- Maximum F1-Score difference: 0.2741
- Average prediction agreement: 0.7212
- Minimum prediction agreement: 0.4475
- Custom implementations need **improvement**