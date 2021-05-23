import os
import numpy as np
from utils import get_r_one_rewards


class Logger:
    def __init__(self, path_to_experiment):
        self.temp_train_delta_rewards = list()
        self.temp_val_delta_rewards = list()
        self.temp_train_greedy_rewards = list()
        self.temp_val_greedy_rewards = list()
        self.temp_train_sample_rewards = list()
        self.temp_val_sample_rewards = list()
        self.path_to_experiment = path_to_experiment
        self.metrics_dir = os.path.join(self.path_to_experiment, "metrics")
        self.model_dir = os.path.join(self.path_to_experiment, "weights")
        self.examples_dir = os.path.join(self.path_to_experiment, "examples")
        self.make_dirs()
        self.train_delta_path = os.path.join(self.metrics_dir, "train_delta.txt")
        self.train_greedy_path = os.path.join(self.metrics_dir, "train_greedy.txt")
        self.train_sample_path = os.path.join(self.metrics_dir, "train_sample.txt")
        self.val_delta_path = os.path.join(self.metrics_dir, "val_delta.txt")
        self.val_greedy_path = os.path.join(self.metrics_dir, "val_greedy.txt")
        self.val_sample_path = os.path.join(self.metrics_dir, "val_sample.txt")

    def log(self, delta, greedy, sample, val):
        if val:
            self.temp_val_delta_rewards += delta.tolist()
            self.temp_val_greedy_rewards.append([greedy['rouge1'][2], greedy['rouge2'][2], greedy['rougeL'][2]])
            self.temp_val_sample_rewards.append([sample['rouge1'][2], sample['rouge2'][2], sample['rougeL'][2]])
        else:
            self.temp_train_delta_rewards += delta.tolist()
            self.temp_train_greedy_rewards.append([greedy['rouge1'][2], greedy['rouge2'][2], greedy['rougeL'][2]])
            self.temp_train_sample_rewards.append([sample['rouge1'][2], sample['rouge2'][2], sample['rougeL'][2]])

    def write_rewards(self, val=False):
        self.write_delta_rewards(val)
        self.write_greedy_rewards(val)
        self.write_sample_rewards(val)

    def write_delta_rewards(self, val=False):
        if val:
            write_str = "%.6f\n" % np.mean(self.temp_val_delta_rewards)
            with open(self.val_delta_path, "a") as f:
                f.write(write_str)
            self.temp_val_delta_rewards = list()
        else:
            write_str = "%.6f\n" % np.mean(self.temp_train_delta_rewards)
            with open(self.train_delta_path, "a") as f:
                f.write(write_str)
            self.temp_train_delta_rewards = list()

    def write_greedy_rewards(self, val=False):
        if val:
            temp = np.array(self.temp_val_greedy_rewards)
            write_str = "%.6f;%.6f;%.6f\n" % (np.mean(temp[:, 0]), np.mean(temp[:, 1]), np.mean(temp[:, 2]))
            with open(self.val_greedy_path, "a") as f:
                f.write(write_str)
            self.temp_val_greedy_rewards = list()
        else:
            temp = np.array(self.temp_train_greedy_rewards)
            write_str = "%.6f;%.6f;%.6f\n" % (np.mean(temp[:, 0]), np.mean(temp[:, 1]), np.mean(temp[:, 2]))
            with open(self.train_greedy_path, "a") as f:
                f.write(write_str)
            self.temp_train_greedy_rewards = list()

    def write_sample_rewards(self, val=False):
        if val:
            temp = np.array(self.temp_val_sample_rewards)
            write_str = "%.6f;%.6f;%.6f\n" % (np.mean(temp[:, 0]), np.mean(temp[:, 1]), np.mean(temp[:, 2]))
            with open(self.val_sample_path, "a") as f:
                f.write(write_str)
            self.temp_val_sample_rewards = list()
        else:
            temp = np.array(self.temp_train_sample_rewards)
            write_str = "%.6f;%.6f;%.6f\n" % (np.mean(temp[:, 0]), np.mean(temp[:, 1]), np.mean(temp[:, 2]))
            with open(self.train_sample_path, "a") as f:
                f.write(write_str)
            self.temp_train_sample_rewards = list()

    def save_model(self, model, epoch, iteration):
        new_model_dir = os.path.join(self.model_dir, "chk_epoch_%d_iteration_%d" % (epoch, iteration))
        if not os.path.isdir(new_model_dir):
            os.mkdir(new_model_dir)
        model.save_pretrained(new_model_dir)

    def save_example(self, article_tokens, abstract_tokens, gt_tokens, tokenizer, iteration):
        article = tokenizer.decode(article_tokens.tolist(), skip_special_tokens=True)
        abstract = tokenizer.decode(abstract_tokens.tolist(), skip_special_tokens=True)
        gt = tokenizer.decode(gt_tokens.tolist(), skip_special_tokens=True)
        _, scores = get_r_one_rewards([gt_tokens], [abstract_tokens], tokenizer)

        path_to_example = os.path.join(self.examples_dir, str(iteration)+'.txt')
        with open(path_to_example, "w") as f:
            delimiter = "\n"*2 + "-"*80 + "\n"*2
            to_write = delimiter.join([article, abstract, gt, str(scores)])
            f.write(to_write)

    def make_dirs(self):
        if not os.path.isdir(self.path_to_experiment):
            os.mkdir(self.path_to_experiment)
        if not os.path.isdir(self.metrics_dir):
            os.mkdir(self.metrics_dir)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.isdir(self.examples_dir):
            os.mkdir(self.examples_dir)
