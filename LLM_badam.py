import math
import os
import torch
import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModelForCausalLM, get_linear_schedule_with_warmup
from custom_datasets import GPTDataset
from badam import BlockOptimizer

import warnings
warnings.filterwarnings("ignore")

class LLM():

    def __init__(self, model_path, load_path="None", source_len=256, cutoff_len=512):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cutoff_len = cutoff_len
        self.source_len = source_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if load_path == "None":
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

        self.model.to(self.device)

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, output_dir,
            do_eval, eval_filename, eval_batch_size):

        train_data = GPTDataset(train_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                    batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        optimizer = BlockOptimizer(
            base_optimizer=optimizer, # can be any torch.Optimizer
            named_parameters_list=list(self.model.named_parameters()), 
            switch_block_every=100, # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter. 
            switch_mode="random", # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
            verbose=2, # information level, will print trainable parameters when setting to 2
            # include_embedding=True,
            # include_lm_head=True,
        )
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_loss = 0, 1e6
        count = 0

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, (input_ids, token_labels) in enumerate(bar):
                input_ids = input_ids.to(self.device)
                labels = token_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                tr_loss += loss.item()
                nb_tr_steps += 1

                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            
            if do_eval:
                # Eval model with dev dataset
                eval_data = GPTDataset(eval_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                print("***** Running evaluation  *****")
                print("  Num examples = %d", eval_data.__len__())
                print("  Batch size = %d", eval_batch_size)
                print("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for (input_ids, token_labels) in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    input_ids = input_ids.to(self.device)
                    labels = token_labels.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                    eval_loss += loss.mean().item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_loss': round(eval_loss, 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)
                if not os.path.exists(output_dir+'/epoch'+str(cur_epoch)):
                    os.makedirs(output_dir+'/epoch'+str(cur_epoch))
                self.model.save_pretrained(output_dir+'/epoch'+str(cur_epoch))
                if best_loss > eval_loss:
                    best_loss = eval_loss
                    print('best eval loss: ', str(eval_loss))
                    count = 0
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    self.model.save_pretrained(output_dir)
                else:
                    count += 1
                    if count == 3:
                        break

                print("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()