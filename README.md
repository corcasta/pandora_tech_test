# Pandora Forecasting Model

A lightweight forecasting service using LitServe and Docker to predict weekly sales for SKUs based on limited historical data.

---

| Path                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `api/`                | API-related code (server endpoint definitions)        |
| `data/`               | Raw and processed datasets                       |
| `images/`             | Visual assets and plots                          |
| `logs/`               | Log files from training or inference             |
| `models/`             | Trained model artifacts                          |
| `notebooks/`          | Jupyter notebooks for exploration and analysis   |
| `scripts/`            | Utility and helper scripts                       |


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
These features are fed to the model for inference. The aggregation level at this point is weekly. Together they help the model learn hidden relations, particularly the sliding window features, they outline clear periodicity and trends in some products. As a result the model can forecast pretty decent.
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

![feature_eng](https://github.com/corcasta/pandora_tech_test/blob/dev/images/fe.png?raw=true)


## 🏗️ Model Architecture
A simple Temporal Convolutional Network (TCN) was used for forecasting because:  
- It performs well with **limited data**.
- It’s effective at detecting temporal patterns.
- It uses dilated causal convolutions to capture dependencies over time.
![tcn](https://github.com/corcasta/pandora_tech_test/blob/dev/images/tcn.png?raw=true)


## 🔌 API Reference
**Note**: The API was intentionally kept simple, with a single route to retrieve forecast information. Currently, a single model has been trained to predict forecasts at the SKU level, which results in a straightforward request format.

To make the process easier for the client, the API accepts a CSV file upload as input. This approach simplifies data handling, but comes with two main considerations:
- **Input requirements**: The uploaded CSV must contain exactly 8 weeks of historical sales data for each product to ensure accurate forecasting.
- **Product identification**: Clients must refer to a predefined list of product IDs to correctly populate the product_id field in the metadata. This mapping ensures clarity about which product is being forecasted when interpreting the results.

These trade-offs were made to balance simplicity on the client side with accuracy and clarity in the forecasting output.
#### Get all items

```http
  POST /predict
```


**Form Data Parameters**

| Name       | Type   | Required | Description                                                                                                    |
| ---------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------- |
| `file`     | file   | Yes      | CSV file containing **exactly 8 rows** of continuous historical weekly sales for **EACH** product. Field name **must** be `file`.               |
| `metadata` | string | Yes      | JSON-encoded string with array describing each SKU in the CSV. See “Metadata JSON” below for schema. |

#### file
The CSV file must contain 8 weeks of historical data for each product. The granularity and the fields of the file must be the exact same as the excel provided in the email.

#### Metadata JSON

The `metadata` field is a JSON string with these keys. The IDs must be the same as the ones mentioned at the beginning.

```json
"metadata": {
  "product_id": [0],
}
```
Example when requesting a forecast for 3 products

```json
"metadata": {
  "product_id": [2, 5, 8],
}
```

Python API request example
```python
url = "http://localhost:8000/predict"
data = {
    'metadata': json.dumps({
        "product_id": [0]
    })
}
files = {
    "file": (
        "data.csv",           
        open("/home/corcasta/projects/pandora_interview/data/sales_data.csv", "rb"), 
        "text/csv"              
    )
}
response = requests.post(url, data=data, files=files)
```
Response example format:
```json
{'1': {'week_0': '61.450638',
  'week_1': '109.70733',
  'week_2': '61.97812',
  'week_3': '0.0'},

'2': {'week_0': '61.450638',
  'week_1': '109.70733',
  'week_2': '61.97812',
  'week_3': '0.0'}}
```

## 🧩 Scaling to Production
The beauty of using LitServe lies in how effortlessly it scales in production. You can simply define the number of workers and specify how many model instances to run and on which devices (CPU or GPU). Distributed execution across multiple machines is also supported just enable the appropriate settings.

**LitServe** also allows flexible request processing strategies: you can choose between batching, parallel execution, or streaming, depending on your needs.

The optimal approach to scaling a forecasting model depends on several factors, such as:

- Who will be consuming the model's predictions
- How much data will be sent per request
- What resources are available for deployment

It's also important to consider operational constraints, such as hardware limitations, budget, and whether the model needs to run continuously or only at scheduled intervals.

The following example scales LitServe to eight parallel workers on a 4 GPU machine, with each of the 4 GPUs serving two copies of the model.
Is as simple as that.
```python
server = ls.LitServer(api, devices=4, workers_per_device=2)
```
For more information I recommend to visit the official page of **LitServe** :)

## 🛠️ Improvements
- Currently, the trained TCN model does not support probabilistic forecasting. In forecasting, having upper and lower bounds is beneficial, as it reinforces confidence that the model operates within a certain range. The downside of implementing this is that training multiple deep learning models increases both training time and computational resource usage.


- The current API only predicts demand for individual SKUs. A secondary endpoint could be added to return aggregated forecasts by category.

- Add flexibility to specify how many weeks into the future to forecast.

- Expand the library of available models and allow users to select a preferred model for forecasting.

- Implement integration tests. Due to personal time constraints, this will be addressed in the future.
