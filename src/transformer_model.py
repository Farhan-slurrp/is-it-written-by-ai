from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import evaluate
import numpy as np
from src.config import Config
from src.data_loader import preprocess_transformer

cfg = Config()



def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels)
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

def train_transformer(df_train, df_val, config=cfg):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = preprocess_transformer(df_train, tokenizer, config.max_length)
    val_dataset = preprocess_transformer(df_val, tokenizer, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(config.model_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(config.log_dir),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        seed=config.seed,
        learning_rate=config.learning_rate,
        save_steps=500,
        report_to="none"  # Disable WandB or other reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if config.save_model:
        trainer.save_model(str(config.model_dir))
        tokenizer.save_pretrained(str(config.model_dir))

    return model, tokenizer, trainer