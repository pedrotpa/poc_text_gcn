from cdaa_utils import *
from process_input import *
from utils import *
from process_graph import *
from model import *

import os
import json
from argparse import ArgumentParser


import torch.optim as optim
def train(A_hat, train_idx, test_idx, train_labels, test_labels, args, device='cpu'):
    X = np.eye(A_hat.shape[0],dtype='float32')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/problem_{args.problem_n}_model_{args.model_no}')

    layers=[args.num_classes]
    if args.hidden_size_2>0:
        layers=[args.hidden_size_2]+layers
    if args.hidden_size_1>0:
        layers=[args.hidden_size_1]+layers
    net = Full_gcn(A_hat, layers, device = device)
    print(net)
    f=torch.from_numpy(np.copy(X)).to(device).float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.77)
    net=net.to(device)

    start_epoch, best_pred = 0, 0
    best_pred=0
    logger.info("Starting training process...")
    info={}
    for e in range(start_epoch, args.num_epochs):
        net.train()
        optimizer.zero_grad()
        output = net(f)
        loss = criterion(output[train_idx], torch.tensor(train_labels, device=device).long() -1)
        loss.backward()
        optimizer.step()
        if (e+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                pred_labels = net(f)
                trained_accuracy = evaluate(output[train_idx], train_labels)
                untrained_accuracy = evaluate(pred_labels[test_idx], test_labels)
            
            writer.add_scalar('training_loss',loss.item(), e)
            writer.add_scalar('trained_accuracy',trained_accuracy, e)
            writer.add_scalar('test_accuracy',untrained_accuracy, e)
            if untrained_accuracy > best_pred:
                best_pred = untrained_accuracy
                torch.save(net.state_dict(), f'runs/problem_{args.problem_n}_bm_{args.model_no}.pt')
                info['epoch']=e
                info['trained_accuracy']=trained_accuracy
                info['test_accuracy']=untrained_accuracy
                info['args']=args
                info['preds']=((pred_labels[test_idx]).max(1)[1]).cpu().numpy()

    save_as_pickle(f"runs/problem_{args.problem_n}_bm_{args.model_no}.info", info)
    print(best_pred,args.problem_n, args.model_no,args.hidden_size_1, args.hidden_size_2, args.lr)
    writer.close()
    del net
    del f
    del optimizer
    del criterion
    del scheduler
    torch.cuda.empty_cache()
    return info['preds']

def main():
    import time
    parser = ArgumentParser()
    parser.add_argument("-i", type=str, help='Path to dataset directory')
    parser.add_argument("--hidden_size_1", type=int, default=32, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=16, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of prediction classes")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    parser.add_argument("--num_epochs", type=int, default=1000, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    parser.add_argument("--problem_n", type=int, default=0, help="problem number")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    args = parser.parse_args()
    USE_GPU = True
    if USE_GPU:
        print(torch.cuda.get_device_name(0))
    device = torch.device('cuda:0' if torch.cuda.is_available() and USE_GPU else 'cpu')
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
    dataset_infos= get_infos(args.i)
    problem = dataset_infos.problem_list[args.problem_n]
    if not os.path.isfile(problem.path+"/preprocessed.pkl"):
        print(f"@@{problem.name}@@")
        start = time.time()
        language = {'en':"english", 'sp':"spanish", 'it':"italian", 'fr':"french"}[problem.language]
        df_data, train_idx, test_idx, train_labels, test_labels = pd_load_problem_data(problem)
        A_hat, node_names = process_graph(df_data['x'], language = language)
        save_as_pickle(problem.path+"/preprocessed.pkl",(A_hat, node_names, train_idx, test_idx, train_labels, test_labels))
        print("Demorou: ",time.time() - start)
    
    A_hat, node_names, train_idx, test_idx, train_labels, test_labels = load_pickle(problem.path+"/preprocessed.pkl")
    train_labels = list(map(lambda z: int(z[-1]),train_labels))
    test_labels = list(map(lambda z: int(z[-1]),test_labels))

    start = time.time()
    preds = train(A_hat, train_idx, test_idx, train_labels, test_labels, args, device=device)
    print(time.time() - start)

    preds = preds+1
    save_as_pickle('preds_'+problem.name+'_m'+str(args.model_no)+'.pkl',preds)

    answers=[]
    idx=0
    for uk in problem.uk_info:
        if (uk[1])!="<UNK>":
            answers.append({'unknown-text':uk[0], 'predicted-author':f"candidate{(preds[idx]):05}"})
            idx+=1

    if not os.path.exists('m_'+str(args.model_no)):
        os.makedirs('m_'+str(args.model_no))
    with open('m_'+str(args.model_no)+'/answers-'+problem.name+'.json', 'w') as f:
        json.dump(answers, f, indent=4)
    
if __name__ == "__main__":
    main()
