## Think-beyond-Instances

## ⚙️ Requirements
You can run pip install -r requirements.txt to deploy the environment.

## ⚙️ Data Preparation
1.  **Data Splitting:** In the experiments, we maintain the same data splitting scheme as the benchmarks.
2.  **Dataset:**  For all datasets (ACM23,AIME25 and so on), please download from the official source.

 ### 📂 Data Preparation

AdaR expects data in **JSONL format**, with each line as:

```json
{
  "query": "The math problem statement",
  "chosen": "The chain-of-thought reasoning",
  "answer": "The gold standard answer"
}
```


## 🔧 Evaluation

To evaluate the predicted answer, run the following command:

```bash
cd code
python show_result.py
