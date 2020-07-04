import torch
import numpy as np
import pickle as pkl
import pandas as pd

movies = pd.read_csv('./scrapy/movies.csv')
device = torch.device('cpu')
model = torch.load('./model.pt', map_location=device)
if isinstance(model,torch.nn.DataParallel):
    model = model.module
Items = np.load('items.npy').reshape(9742, 3, 12)
item2id = pkl.load(open('item2id.pickle', 'rb'))
id2item = pkl.load(open('id2item.pickle', 'rb'))
genres2id = {g:i for i, g in enumerate("Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western,(no genres listed),IMAX".split(','))}

def modelPred(collection=[59369, 45880, 44511, -1, -1, 82041, -1, -1]):
    hist = []
    genre = []
    for mid in collection:
        if mid < 0:
            continue
        else:
            mg = movies[movies['movieId']==mid]['genres'].values[0].split('|')
            genre.extend([genres2id[g] for g in mg])
            hist.append(item2id['%d'%mid])
    genre = list(set(genre))
    cntx = np.zeros((3, 1 + len(hist) + len(genre)), dtype=np.float32)
    cntx[0, 1:(1+len(hist))] = 4
    cntx[1, 1:(1+len(hist))] = np.array(hist)
    cntx[2, 1:(1+len(hist))] = 1./len(hist)
    cntx[0, (1+len(hist)):] = 5
    cntx[1, (1+len(hist)):] = np.array(genre)
    cntx[2, (1+len(hist)):] = 1./len(genre)
        
    context = torch.tensor(np.tile(cntx, (Items.shape[0], 1, 1))).to(device)
    items = torch.tensor(Items).to(device)
    model.eval()
    with torch.no_grad():
        y = model(context[:, 0, :].to(torch.long), \
                context[:, 1, :].to(torch.long), \
                context[:, 2, :], \
                items[:, 0, :].to(torch.long), \
                items[:, 1, :].to(torch.long), \
                items[:, 2, :])
        predicts = torch.flatten(y.detach()).cpu().numpy()
        sorted_id = np.argsort(predicts)[::-1][:5]
        sorted_id = [id2item[i] for i in sorted_id]
    return sorted_id

modelPred()

