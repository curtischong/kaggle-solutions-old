- [https://medium.com/ether-labs/bert-for-unsupervised-text-tasks-fa6e9ce5d133](https://medium.com/ether-labs/bert-for-unsupervised-text-tasks-fa6e9ce5d133)
- idea 1: create a document embedding then use similarity metrics (e.g. cosine similarity)
    - problem: traditional similarity metrics doesn't work too well
- idea 2: use the next sentence prediction (NSP) that BERT has
    - you concat the second sentence to the end of the first one
    - Then you see the NSP probability of those sentences
- Idea 3: extend NSP to context window
    - the idea is that similar sentences are near each other
    - so you do NSP on the current sentence and +- 2 sentences away
        - this requires more training
- We typically use a weighted combination of cosine similarity and context window score to measure sentence relationship

- problem: it's hard to create an embedding for an large documents
- solution: select 1 sentence from each document to represent the entire document. create embeddings for this 1 document

- Once we have the similarity score between sentences we can run kmeans. The text segments closest to each centroid becomes the document embedding candidate
- OR
- we can use community detection algorithms (e.g. [[Louvain Algorithm]])
- then use graph metrics such as node/edge centrality, PageRank to identify the influential node in each sub-graph
    - the node with the highest metric becomes the document embedding candidate