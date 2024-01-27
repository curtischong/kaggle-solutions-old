- https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting/discussion/271890
- when teaming with other people late, everyone will have their own implementation. Rather than trying to merge all the code into one monolithic codebase, it seems easier to just make a class for each person's code, and call each one's functions individually:

```python
data = read_data()

c1 = Class1()
c2 = Class2()
c3 = Class3()

# cumulative features (targetLag, yesterdayScore, sinceLastGames, ..)
data1 = c1.preprocess(data)
data2 = c2.preprocess(data)
data3 = c3.preprocess(data)

for test_df, sample in iter_test:
    p1, data1 = c1.pred(test_df, sample, data1)
    p2, data2 = c2.pred(test_df, sample, data2)
    p3, data3 = c3.pred(test_df, sample, data3)

    p_final = p1 * x1 p2 * x2 + p3 * x3
```