#pragma once

#include <vector>
#include <deque>
#include <map>
#include <string>
#include <memory>
#include <NvInfer.h>

namespace atos {
namespace analytics {

/**
 * @brief World-Class Predictive Traffic Forecasting Engine.
 * 
 * Uses a Time-Series Model (LSTM or TCN) to predict future traffic 
 * volume and congestion states based on historical graph metrics.
 */
class ForecastingEngine {
public:
    struct Prediction {
        float predictedDensity;
        float confidence;
        std::string congestionLevel; 
    };

    ForecastingEngine(int historyWindow = 60, int predictionHorizon = 15);

    /**
     * @brief Update the model with current density data.
     */
    void update(int nodeId, float currentDensity);

    /**
     * @brief Predict future state for a specific camera node.
     */
    Prediction predict(int nodeId);

private:
    int history_window;       
    int prediction_horizon;   
    
    // NodeId -> Historical Sequence
    std::map<int, std::deque<float>> historical_data;

    // TensorRT Engine for LSTM/TCN Inference
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    float runInference(const std::vector<float>& sequence);
};

} // namespace analytics
} // namespace atos
