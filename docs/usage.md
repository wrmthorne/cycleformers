## 🚧 Usage documentation still under construction

### Getting the Best Performance

#### Tokenization

Use models with a compatible tokenizer e.g. the same model or a smaller and larger version of the same model. Sending tokens from the GPU, back to CPU to detokenize, manipulate as strings, then re-tokenizing and sending back to GPU is costly. If the same tokenizer is applicable for both models, the tokens can be manipulated while still as tensors on the GPU, reducing overhead.

However, this cost may be unavoidable such as in situations where chat templates are used.
