# Apply Lightweight Fine-Tuning to a Foundation Model

Lightweight fine-tuning is one of the most important techniques for adapting foundation models, because it allows us to modify foundation models for our needs without needing substantial computational resources.
## Project Summary
In this project, we will bring together all of the essential components of a PyTorch + Hugging Face training and inference process. Specifically, we will:

1) Load a pre-trained model and evaluate its performance
2) Perform parameter-efficient fine tuning using the pre-trained model
3) Perform inference using the fine-tuned model and compare its performance to the original model

## Project Details
In this project, we fine tune [DistilBERT base model (uncased)](https://huggingface.co/distilbert/distilbert-base-uncased) through PEFT method of [Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). DistilBERT was chosen as it is a comparatively lightweight model and requires less computational resources. LoRA was chosen as the PEFT method as it one of the most effective and widely utilized method.
[IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) was chosen for finetuning. This dataset is for binary sentiment classification of movie reviews.
Two different configurations of LoRA were analyzed by changing the rank because it is the one of the most significant LoRA parameter and directly impacts the number of trainable parameters as well. The results are provided in the table below.

|           Model           | Trainable Parameters | Evaluation Accuracy |
|:-------------------------:|:--------------------:|:-------------------:|
|          Original         |          N/A         |        52.6%        |
|  LoRA (rank=8) fine-tuned |         1.22%        |        83.2%        |
| LoRA (rank=16) fine-tuned |         1.54%        |        82.0%        |

The [conda environment file](./GenAI.yml) is provided for installing the relevant packages needed to run the [notebook](./LightweightFineTuning.ipynb).

### Observations
- For both LoRA configurations, validation loss starts to increase after 2 epochs ('test' split was utilized for validation).
- Evaluation accuracy on the test dataset is slightly different upon loading the fine-tuned model weights as compared to evaluation performed via trainer right after fine-tuning. In the table above, the metrics obtained after performing inference on loaded model weights are provided.
- Seed was set prior to initializing the base model to make the training reproducible.

## Conclusion
LoRA is a very efficient way (training only around 1% of base model parameters in this case) to fine-tune a base model and significantly increase (>=30% in this case) its performance on downstream tasks.