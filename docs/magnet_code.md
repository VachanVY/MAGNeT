# Magnet Code Docs

## Magnet Model Docs
### Why is the Input and Output Embedding cardinality different?
```python
...
self.embeddings = nn.ModuleList([
    nn.Embedding(
        # something like vocab_size
        num_embeddings=config.cardinality + 1, # +1 for mask_id
        embedding_dim=config.d_model
    ) for _ in range(self.nq)
]) # 
self.linears = nn.ModuleList([
    nn.Linear(
        config.d_model, config.cardinality
    ) for _ in range(self.nq)
])
...
```
```python
Input:  [100, 120, 20, 200, 1024, 1024, 1000, 1024]
# model will predict the masked values, so no need of the masked 
Output: [100, 120, 20, 200, <predValue>, <predValue>, 1000, <predValue>]
# <predValue> is the value predicted by the model 
```
* $predValue  \in [0, 1023]$


## Explanation for `Binary AND` for loss mask
```python
>>> loss_mask = torch.tensor(
    [[True, True, False, False, False, True, True],
        [False, True, True, False, False, False, True]]
) # True values to be taken for loss, False values to be ignored

>>> pad_mask = torch.tensor(
    [[True, True, True, False, False, False, False],
        [True, True, True, True, True, False, False]]
)
>>> loss_mask & pad_mask # take loss only on masked tokens (True value)
tensor([[ True,  True, False, False, False, False, False],
        [False,  True,  True, False, False, False, False]])
```