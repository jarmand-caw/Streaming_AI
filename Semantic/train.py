from Semantic.genre_lstm import GenreEngine
from Semantic.utils import PreprocessDataFrame
import pandas as pd

META_DATA_PATH = '/Users/jackarmand/PycharmProjects/Streaming_AI/Semantic/movies_metadata.csv'
GOOGLE_W2V_VECTOR_PATH = '/Users/jackarmand/PycharmProjects/Streaming_AI/Semantic/GoogleNews-vectors-negative300.bin.gz'
df = pd.read_csv(META_DATA_PATH)
pdf = PreprocessDataFrame(df)
X,y = pdf.process(GOOGLE_W2V_VECTOR_PATH)

config = {'vocab_size':len(pdf.word_index)+1,'n_out':y.shape[1],'n_layers':2,'n_hidden':64,
          'w2v_weights_matrix':pdf.w2v_embedding,'use_cuda':False,
          'model_dir':'/content/drive/My Drive/Movie_Categorization/Models/',
          'optimizer':'adam','adam_lr':1e-2,'l2_regularization':0,'lstm':True,'clip':5}

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=1)

train_dataset = TensorDataset(torch.LongTensor(X_train),torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.LongTensor(X_test),torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=200,drop_last=True)

for i,(text,label) in enumerate(train_loader):
    print(text.shape,label.shape)
    if i==0:
        break

epochs = 30
engine = GenreEngine(config)
best_score=1e10
for epoch in range(epochs):
    engine.train_an_epoch(train_loader,epoch)
    score = engine.evaluate(test_loader,epoch)
    if score<best_score:
        engine.save(config['model_name'], (epoch+1), score)

