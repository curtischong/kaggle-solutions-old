- For each query, the model gives you a list of proposed results
- The reciprocal rank is the reciprocal of the rank of the first valid result
- MRR is the mean of the Reciprocal Rank of the queries:

| Query | Proposed Results | Correct response | Rank | Reciprocal rank |
| ---- | ---- | ---- | ---- | ---- |
| cat | catten, cati, **cats** | **cats** | 3 | 1/3 |
| torus | torii, **tori**, toruses | **tori** | 2 | 1/2 |
| virus | **viruses**, virii, viri | **viruses** | 1 | 1 |
MRR = (1/3 + 1/2 + 1)/3 = 11/18

-  Note: If the model returns multiple correct answers for a query, it only cares about the rank of the **first** correct answer