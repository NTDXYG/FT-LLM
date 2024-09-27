from nlp2 import set_seed
from LLM_badam import LLM
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
set_seed(42)

def train(model_path):
    lm_model = LLM(model_path=model_path, load_path="None", source_len=128, cutoff_len=256)

    lm_model.train(train_filename='./dataset/CodeHarmony_train.jsonl', train_batch_size=2, learning_rate=2e-4, 
                        num_train_epochs=5, do_eval=True, eval_filename='./dataset/CodeHarmony_valid.jsonl', 
                        eval_batch_size=1, output_dir='./save_models_badam/')

train('D:\models\CodeGPT-small-py-adaptedGPT2')