from cdaa_utils import *
from process_input import *
from utils import *
from process_graph import *
from model import *

import os
from argparse import ArgumentParser

def load_bible(args, test_ratio=0.1):
    logger.info("Preparing data...")
    df = pd.read_csv(os.path.join(args.i,"t_bbe.csv"))
    df.drop(["id", "v"], axis=1, inplace=True)
    df = df[["t","c","b"]]

    df_data = pd.DataFrame(columns=["c", "b"])
    for book in df["b"].unique():
        dum = pd.DataFrame(columns=["c", "b"])
        dum["c"] = df[df["b"] == book].groupby("c").apply(lambda x: (" ".join(x["t"])).lower())
        dum["b"] = book
        df_data = pd.concat([df_data,dum], ignore_index=True)
    del df
    df_data = df_data.rename(columns={'c':'x', 'b':'y'})

    test_idx = []
    for b_id in df_data["y"].unique():
        dum = df_data[df_data["y"] == b_id]
        if len(dum) >= 4:
            test_idx.extend(list(np.random.choice(dum.index, size=round(test_ratio*len(dum)), replace=False)))
    test_labels = [l for idx, l in enumerate(df_data["y"]) if idx in test_idx]
    
    train_idx = []
    for i in range(len(df_data)):
        if i not in test_idx:
            train_idx.append(i)
    train_labels = [l for idx, l in enumerate(df_data["y"]) if idx in train_idx]
    return df_data, train_idx, test_idx,train_labels, test_labels

def preprocess_bible(args):
    import time
    print(f"@@bible@@")
    start = time.time()
    df_data, train_idx, test_idx, train_labels, test_labels = load_bible(args, 0.1)
    A_hat, node_names = process_graph(df_data['x'], language = "english")
    
    save_as_pickle(os.path.join(args.i,"preprocessed.pkl"),(A_hat, node_names, train_idx, test_idx, train_labels, test_labels))
    print("Demorou: ",time.time() - start)

import torch.optim as optim
def train(A_hat, train_idx, test_idx, train_labels, test_labels, args, device='cpu'):
    X = np.eye(A_hat.shape[0],dtype='float32')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/model_{args.model_no}')

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
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f\t best: %.7f" % (e, untrained_accuracy, best_pred))
            if untrained_accuracy > best_pred:
                best_pred = untrained_accuracy
                info['epoch']=e
                info['trained_accuracy']=trained_accuracy
                info['test_accuracy']=untrained_accuracy
                info['args']=args
                info['preds']=((pred_labels[test_idx]).max(1)[1]).cpu().numpy()

    writer.close()
    del net
    del f
    del optimizer
    del criterion
    del scheduler
    torch.cuda.empty_cache()
    return info['preds']

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", type=str, help='Path to dataset directory')
    parser.add_argument("--hidden_size_1", type=int, default=32, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=16, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=66, help="Number of prediction classes")
    parser.add_argument("--num_epochs", type=int, default=1200, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.i,"preprocessed.pkl")):
        preprocess_bible(args)
    
    import time
    USE_GPU = True
    if USE_GPU:
        print(torch.cuda.get_device_name(0))
    device = torch.device('cuda:0' if torch.cuda.is_available() and USE_GPU else 'cpu')
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
    A_hat, node_names, train_idx, test_idx, train_labels, test_labels = load_pickle(os.path.join(args.i,"preprocessed.pkl"))
    start = time.time()
    preds = train(A_hat, train_idx, test_idx, train_labels, test_labels, args, device=device)
    print(time.time() - start)
    save_as_pickle("preds/bible_preds.pkl", preds)
    
    

if __name__ == "__main__":
    main()