import os
import time
import torch
import argparse
import optuna

#from model import SASRec
from model_v2 import SASRec
#from utils import *
from utils_v2 import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
#args.train_dir='tune_res'
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
#with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
#f.close()

def objective(trial):
  args.lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)
  args.hidden_units=trial.suggest_int('hidden_units', 30, 70, step=10)
  args.dropout_rate= trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
  args.maxlen= trial.suggest_int('maxlen', 50, 150, step=25)
  args.l2_emb=trial.suggest_float('l2_emb', 0.0, 0.0001, step=0.00005)
  dataset = data_partition(args.dataset)
  [user_train, user_valid, user_test, usernum, itemnum] = dataset
  num_batch = len(user_train) // args.batch_size 

  sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
  model = SASRec(usernum, itemnum, args).to(args.device)

  for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
  
  model.train() # enable model training
  epoch_start_idx = 1
  bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
  adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
  T = 0.0
  t0 = time.time()
  for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
        if epoch == args.num_epochs:
            model.eval()
            t1 = time.time() - t0
            T += t1
            t0 = time.time()
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            kpi=t_valid[0]
  return kpi

if __name__ == '__main__':
    # global dataset
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2023))
    study.optimize(objective, n_trials=30)
    print(f'Trial {study.best_trial.number} get the best NDCG@10({study.best_trial.value}) with params: {study.best_trial.params}')
    print("Done")
