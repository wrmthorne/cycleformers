## Cycle-consistency Training

Cycle-consistency training (CCT) creates a closed feedback loop between source and target domains by linking two models together during training. Each model implements the inverse function of the other i.e. $f(g(x)) = x$, for example one that translates  English to German and the other German to English. Both models are trained jointly trained using a single optimiser on the round-trip reconstruction of input data from non-parallel dataset. Quite often, cycle-consistency loss is used as a secondary loss component such as in [Zhu et al. 2017](https://arxiv.org/abs/1703.10593) which also uses adversarial loss.

📈 GRAPH TO BE INSERTED HERE

For tasks such as text generation, we must sample discrete tokens from the model's continuous output distribution, breaking the gradient flow. In these settings, $f(g(x))$ is non-differentiable, preventing us from using the standard CCT loss. While CCT enforces a closed loop within each training batch, iterative back translation (IBT) avoids the same optimization issue by using one model to generate synthetic parallel data for the other to use as input. Each cycle therefore has a separate loss function and optimiser, alternating between training each model ([Gou et al. 2020](https://arxiv.org/abs/2006.04702)).

📈 GRAPH TO BE INSERTED HERE