https://www.kaggle.com/code/shonenkov/wbf-approach-for-ensemble
via GPT: The process of non-maximum suppression includes:

1. **Sorting the detections:** Begin by sorting the detection boxes by their objectness score. The bounding box with the highest score will usually be the most confident one.
    
2. **Selecting the detection with the highest score:** Select the top-score box and eliminate all the boxes which have high Intersection over Union (IoU) with respect to the top-score box. IoU is a technique to measure overlap between 2 boundaries; it's the ratio of the intersection of two boxes to their union.
    
3. **Iterating the earlier steps:** This process is repeated for each of the remaining bounding boxes. If you have n bounding boxes, worst case time complexity of NMS can be O(n^2), because each bounding box can be compared once with every other box.