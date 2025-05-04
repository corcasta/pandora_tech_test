# Pandora Forecasting Model

A lightweight forecasting service using LitServe and Docker to predict weekly sales for SKUs based on limited historical data.

---

## 🔧 Setup Instructions

By following these steps, you will have a LitServe server running and ready for inference:

1. **Clone the repository**

   ```
   git clone git@github.com:corcasta/pandora_tech_test.git
   cd pandora_tech_test
   ```

2. **Build Docker Image**
    ```
    docker build -t pandora-forecasting-model .
    ```

3. **Run the Docker container**
    ```
    docker run -p 8000:8000 pandora-forecasting-model:latest
    ```

4. **Run inference**
    ```
    python inference.py
    ```

## 🧠 Considerations
Based on the provided Excel/data, a few assumptions were made:
- All products within a category that share the same price are assumed to represent the same product or SKU.
- As there were not direct IDs, I defined a local ID for each product.
    | Category     | Price | ID  |
    |--------------|-------|-----|
    | Beauty       | 25    | 0   |
    | Beauty       | 30    | 1   |
    | Beauty       | 50    | 2   |
    | Beauty       | 300   | 3   |
    | Beauty       | 500   | 4   |
    | Clothing     | 25    | 5   |
    | Clothing     | 30    | 6   |
    | Clothing     | 50    | 7   |
    | Clothing     | 300   | 8   |
    | Clothing     | 500   | 9   |
    | Electronics  | 25    | 10  |
    | Electronics  | 30    | 11  |
    | Electronics  | 50    | 12  |
    | Electronics  | 300   | 13  |
    | Electronics  | 500   | 14  |



## 🎯 Upper Level Design Choices
- The model forecasts per SKU.
- It predicts 4 weeks of future sales.
- It requires 8 continuous weeks of historical data to infer the next 4 weeks.

**Note**:
Ideally, for a retailer like Pandora, forecasting should cover 8–16 weeks to enable proper planning by operations and logistics teams. However, due to limited historical data, the forecast horizon was constrained to 4 weeks. This allows for a larger training dataset and demonstrates how the system could operate in a real-world scenario.  

**Also**, weekly sales data was used instead of daily data because:
- This choice helps reduce the impact of outliers and commercial noise.
- Weekly aggregation reveals more meaningful patterns, providing greater business value.

## ⚙️ Feature Engineering
These features are fed to the model for inference. The aggregation level at this point is weekly.
| Feature         | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| Total_Amount    | Total sales (e.g., price × quantity)                               |
| Age             | Median customer’s age in years                                     |
| Male            | Total male count                                                   |
| Female          | Total female count                                                 |
| Quantity        | Number of units purchased in the transaction                       |
| Price_per_Unit  | Price charged per individual unit/item                             |
| Year            | Year when the transaction occurred (e.g., 2025)                    |
| Month           | Month of the transaction (1–12)                                    |
| Week            | ISO week number of the year for the transaction (1–52)             |
| Window_Mean_4   | Rolling average of sales or quantity over the past 4 weeks         |
| Window_Mean_5   | Rolling average of sales or quantity over the past 5 weeks         |
| Window_Mean_6   | Rolling average of sales or quantity over the past 6 weeks         |
| Window_Mean_7   | Rolling average of sales or quantity over the past 7 weeks         |

This is an example:
![fe](![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)
)


## 🏗️ Model Architecture
A simple Temporal Convolutional Network (TCN) was used for forecasting because:  
- It performs well with **limited data**.
- It’s effective at detecting temporal patterns.
- It uses dilated causal convolutions to capture dependencies over time.
