
#! coding=utf-8
import os
from tkinter import *
import tkinter.scrolledtext as tkst
from PIL import Image, ImageTk
import pandas as pd
import random as rm
import torch
import numpy as np
import pickle as pkl
import csv

LARGE_FONT = ("Verdana",20)
BTN_FONT = ("Verdana",14)

img=None
image=None

c_fst_img=None
c_fst_image=None

c_img=None
c_image=None

r_fst_img=None
r_fst_image=None

r_img=None
r_image=None

c_steps = 1
result_id = 0

collection =[]
query = ''

img_path = './scrapy/images/'
result = pd.read_csv('./scrapy/result.csv')
movies = pd.read_csv('./scrapy/movies.csv')

movieId_in_result = set(list(result['movieId']))
movieId_in_movies = set(list(movies['movieId']))
movieId = list((movieId_in_result.intersection(movieId_in_movies)))

result = result[result['movieId'].isin(movieId)].reset_index(drop=True)
movies = movies[movies['movieId'].isin(movieId)].reset_index(drop=True)

idx = list(result.index.values)

# yaxu
device = torch.device('cpu')
model = torch.load('./model.pt', map_location=device)
if isinstance(model,torch.nn.DataParallel):
    model = model.module
Items = np.load('items.npy').reshape(9742, 3, 12)
item2id = pkl.load(open('item2id.pickle', 'rb'))
id2item = pkl.load(open('id2item.pickle', 'rb'))
genres2id = {g:i for i, g in enumerate("Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western,(no genres listed),IMAX".split(','))}
# yaxu

def getMovieName(sample_id):
    name = []
    for i in sample_id:
        mn = movies[movies['movieId']==i]['title'].values[0]
        name.append(mn)
    return name

def getMovieGenre(sample_id):
    genre = []
    for i in sample_id:
        mg = movies[movies['movieId']==i]['genres'].values[0]
        genre.append(mg)
    return genre

def modelPred():
    hist = []
    genre = []
    print(collection+h_sample_movieId)
    for mid in (collection+h_sample_movieId):
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
        sorted_id = [int(id2item[i]) for i in sorted_id]
    return sorted_id

def chenghung_function(query):
    query = (query.lower()).split()
    if len(query) >= 1:
        query = query[-1]
    else:
        query = 'romance'

    with open('movies_correspond_category_and_director.csv', newline='', encoding="utf-8") as resultfile:
        rows = csv.reader(resultfile, delimiter=',') 
        movieID = []
        for row in rows:
            if query == row[0]:
                movieID += list(np.array(row[1].split(' ')[:5], dtype = int))
                break
    return movieID 

'''
c: Collection for recommender
'''
c_sample = rm.sample(idx,9)
c_sample_movieId = list(result.iloc[c_sample]['movieId'])
c_movie_name = getMovieName(c_sample_movieId)
c_movie_genre = getMovieGenre(c_sample_movieId)
c_img_lst = [img_path+str(x)+'.jpg' for x in c_sample_movieId]


'''
r_sample should be search result
'''
r_sample = rm.sample(idx,9)
r_sample_movieId = list(result.iloc[r_sample]['movieId'])
r_movie_name = getMovieName(r_sample_movieId)
r_movie_genre = getMovieGenre(r_sample_movieId)
r_img_lst = [img_path+str(x)+'.jpg' for x in r_sample_movieId]

'''
more hist
'''
h_sample = rm.sample(idx, 50)
h_sample_movieId = list(result.iloc[h_sample]['movieId'])

overview = result[result['movieId'].isin(r_sample_movieId) ]
overview = list(overview['overview'])

class Win(Tk):
    def __init__(self, master, *args, **kwargs):
        self.master = master

        master.title("Movie Finder")
        master.geometry('1200x800+400+10')
        master.config(bg='#323232')

        container = Frame(master, width=1000, height=800, relief='flat')
        container.pack(expand=True, fill='both')

        container.pack_propagate(0)
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)

        self.frames={}
        for F in (StartPage, Collection, Search, Result):
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.config(bg='#323232')
        self.show_frame(StartPage)

    def show_frame(self, cont):
        if cont == Result:
            self.frames[cont].reinit()
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(Frame):
    def __init__(self, parent, controller):
        global img
        global image
        global c_img_lst
        Frame.__init__(self, parent)
        title = Label(self,fg='white',bg='#323232',text='Welcome to Movie Finder \n please press Start!')
        title.config(font=LARGE_FONT)
        title.pack(pady=(50,30),padx=(20,0))

        img =  Image.open(c_img_lst[0]).resize((400,500))
        img = ImageTk.PhotoImage(img)
        image = Label(self, image = img)
        image.pack()
        btn = Button(self, text='Start!',
                       command=lambda:controller.show_frame(Collection))
        btn.config(font=LARGE_FONT)
        btn.pack(pady=(0,30),side = 'bottom')

class Collection(Frame):
    def __init__(self, parent, controller):
        global c_fst_img
        global c_fst_image

        global c_img_lst
        global c_movie_name
        global c_movie_genre

        global collection
        global c_sample_movieId

        Frame.__init__(self, parent)

        title = Label(self,fg='white',bg='#323232',text='Please choose Watch or Unwatch')
        title.config(font=LARGE_FONT)
        title.pack(pady=(50,30),padx=(20,0))

        c_fst_img =  Image.open(c_img_lst[1]).resize((400,500))
        c_fst_img = ImageTk.PhotoImage(c_fst_img)
        c_fst_image = Label(self, image = c_fst_img)
        c_fst_image.pack()

        name = Label(self,fg='white',bg='#323232',text=c_movie_name[1])
        name.pack()

        '''
        get checkbox value
        '''
        CheckVar1 = IntVar()
        CheckVar2 = IntVar()
        C1 = Checkbutton(self, text = " Watch ", variable = CheckVar1, selectcolor="#FFDCB9",
                 onvalue = c_sample_movieId[c_steps], offvalue = 0, font=BTN_FONT,indicatoron=False, width=8,
                         command=lambda:self.getC1(CheckVar1))

        C2 = Checkbutton(self, text = " Unwatch ", variable = CheckVar2, selectcolor="#FFDCB9",
                 onvalue = -1, offvalue = 0, font=BTN_FONT,indicatoron=False, width=8,
                        command=lambda:self.getC2(CheckVar2))


        C1.place(relx = 0.4,rely = 0.9,anchor = CENTER)
        C2.place(relx = 0.6,rely = 0.9,anchor = CENTER)

        btn = Button(self,command=lambda:self.ChangeImg(c_movie_name,c_img_lst,c_sample_movieId,name,C1,C2,btn,controller))
        btn.config(text=' Next ',font=BTN_FONT, width=8)
        btn.place(relx = 0.8,rely = 0.9,anchor = CENTER)

    def getC1(self,CheckVar1):
        collection.append(CheckVar1.get())

    def getC2(self,CheckVar2):
        collection.append(CheckVar2.get())

    def ChangeImg(self,c_movie_name,c_img_lst,c_sample_movieId,name,C1,C2,btn,controller):
        global c_img
        global c_image
        global c_steps
        c_steps += 1

        C1.destroy()
        C2.destroy()

        CheckVar1 = IntVar()
        CheckVar2 = IntVar()
        C1 = Checkbutton(self, text = " Watch ", variable = CheckVar1, selectcolor="#FFDCB9",
                 onvalue = c_sample_movieId[c_steps], offvalue = 0, font=BTN_FONT,indicatoron=False, width=8,
                         command=lambda:self.getC1(CheckVar1))


        C2 = Checkbutton(self, text = " Unwatch ", variable = CheckVar2, selectcolor="#FFDCB9",
                 onvalue = -1, offvalue = 0, font=BTN_FONT,indicatoron=False, width=8,
                        command=lambda:self.getC2(CheckVar2))


        C1.place(relx = 0.4,rely = 0.9,anchor = CENTER)
        C2.place(relx = 0.6,rely = 0.9,anchor = CENTER)

        c_img = Image.open(c_img_lst[c_steps]).resize((400,500))
        c_img = ImageTk.PhotoImage(c_img)
        c_fst_image.config(image=c_img)
        name.config(text=c_movie_name[c_steps])

        if c_steps == 8:
            btn.config(text='Go to search',font=BTN_FONT, width=15,
                       command=lambda:controller.show_frame(Search))


class Search(Frame):
    def __init__(self,parent,controller):
        global query
        Frame.__init__(self, parent)
        title = Label(self,bg='#323232',fg='white',
                      text='Please enter one or two key words in English\nWhen you finish typing, please press OK! first')
        title.config(font=LARGE_FONT)
        title.pack(pady=(50,30),padx=(20,0))

        search =  Entry(self, font=LARGE_FONT, bd=5)
        search.pack()

        btn = Button(self, text='OK!',bd=3,command=lambda:self.getQuery(search))
        btn.config(font=LARGE_FONT)
        btn.pack(pady=(20,0))

        btn = Button(self, text='Then Go Search!',bd=3,command=lambda:controller.show_frame(Result))
        btn.config(font=LARGE_FONT)
        btn.pack(pady=(20,0))

    def getQuery(self,search):
        global query
        query = search.get()


class Result(Frame):
    def __init__(self,parent,controller):
        Frame.__init__(self, parent)

        title = Label(self,bg='#323232',fg='white',text='Movies you may like')
        title.config(font=LARGE_FONT)
        title.pack(pady=(50,30),padx=(20,0))

    def reinit(self):
        global r_fst_img
        global r_fst_image
        global r_img_lst
        global overview
        global result_id
        global r_movie_name
        global r_movie_genre
        global query

        frame_l = Frame(self,bd=5)
        frame_l.pack(side='left')

        frame_r = Frame(self,bg='#323232')
        frame_r.pack(side='right',expand=True, fill='both')

        frame_title = Frame(frame_r)
        frame_title.pack(expand=True, fill='both')

        frame_r_t = Frame(frame_r)
        frame_r_t.pack()

        frame_r_b = Frame(frame_r,bd=3)
        frame_r_b.pack()

        lb = Label(frame_l,text="Recommended Movies")
        lb.pack()
        movie_lst = Text(frame_l,height=400,width=30,bg='#323232')
        movie_lst.config(font=("微软雅黑",12),fg='white',bg='#323232')
        movie_lst.pack()
        '''
        recommender result
        '''
        yaxu_movie_id = modelPred()
        yaxu_movie_name = getMovieName(yaxu_movie_id)
        for r in yaxu_movie_name:
            movie_lst.insert(END,'\n')
            movie_lst.insert(INSERT,r)
            movie_lst.insert(END,'\n')

        poster = Label(frame_title,fg='black',text="Search Result")
        poster.pack()

        r_sample_movieId = chenghung_function(query)
        r_movie_name = getMovieName(r_sample_movieId)
        r_movie_genre = getMovieGenre(r_sample_movieId)
        r_img_lst = [img_path+str(x)+'.jpg' for x in r_sample_movieId]
        overview = result[result['movieId'].isin(r_sample_movieId) ]
        overview = list(overview['overview'])

        r_fst_img =  Image.open(r_img_lst[result_id]).resize((350,450))
        r_fst_img = ImageTk.PhotoImage(r_fst_img)
        r_fst_image = Label(frame_r_t, image = r_fst_img)
        r_fst_image.config(bg='#323232')
        r_fst_image.pack(expand=True, fill='both')

        movie_name = Label(frame_r_t, text='Name:  '+r_movie_name[result_id],font=("微软雅黑",12),fg='white',bg='#323232')
        movie_genre = Label(frame_r_t, text='Genre:  '+r_movie_genre[result_id],font=("微软雅黑",12),fg='white',bg='#323232')
        movie_name.pack(expand=True, fill='both')
        movie_genre.pack(expand=True, fill='both')

        intro = Text(frame_r_b,font=("微软雅黑",14),fg='white',bg='#323232')
        intro.pack(expand=True, fill='both')
        intro.insert(INSERT,'Overview:  '+ overview[result_id])

        preButton = Button(frame_r,command=lambda:self.preMovie(r_img_lst,movie_name,movie_genre,intro))
        preButton.config(text=' Previous ',font=BTN_FONT, width=8)
        preButton.place(anchor = CENTER, relx=0.15,rely=0.5)

        nextButton = Button(frame_r,command=lambda:self.nextMovie(r_img_lst,movie_name,movie_genre,intro))
        nextButton.config(text=' Next ',font=BTN_FONT, width=8)
        nextButton.place(anchor = CENTER, relx=0.85,rely=0.5)


    def nextMovie(self,r_img_lst,movie_name,movie_genre,intro):
        global r_img
        global r_image
        global result_id
        global overview
        global r_movie_name
        global r_movie_genre

        result_id += 1
        if result_id < len(r_img_lst):
            r_img = Image.open(r_img_lst[result_id]).resize((350,450))
            r_img = ImageTk.PhotoImage(r_img)
            r_fst_image.config(image=r_img)

            movie_name.config(text='Name:  '+r_movie_name[result_id])
            movie_genre.config(text='Genre:  '+r_movie_genre[result_id])

            intro.delete(1.0,END)
            intro.insert(INSERT,'Overview:  '+ overview[result_id])
        else:
            result_id -= 1

    def preMovie(self,r_img_lst,movie_name,movie_genre,intro):
        global r_img
        global r_image
        global result_id
        global overview
        global r_movie_name
        global r_movie_genre
        if result_id > 0 and result_id < len(r_img_lst):
            result_id -= 1
            r_img = Image.open(r_img_lst[result_id]).resize((350,450))
            r_img = ImageTk.PhotoImage(r_img)
            r_fst_image.config(image=r_img)

            movie_name.config(text='Name:  '+ r_movie_name[result_id])
            movie_genre.config(text='Genre:  '+ r_movie_genre[result_id])

            intro.delete(1.0,END)
            intro.insert(INSERT,'Overview:  '+ overview[result_id])

if __name__ == '__main__':
    root = Tk()
    root.attributes('-topmost', True) #视窗置顶
    app = Win(root)
    root.mainloop()

print('recommender input is',collection)
print('search query is', query)
