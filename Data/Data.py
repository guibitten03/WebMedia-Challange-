import pickle as pkl
from torch.utils.data import Dataset


class NewsData():
    def __init__(self, root_path, data_mode='Val', rating_mode="time", user_doc_mode=""):

        '''
            Classe para carregar os arquivos pkl.
        '''

        self.source_path = root_path
        self.data_mode = data_mode
        self.user_doc_mode = user_doc_mode
        self.load()


        # self.data = [self.ratingMatrix[i][0] for i in range(len(self.ratingMatrix))]
        # if rating_mode == "time":
        #     self.scores = [self.ratingMatrix[i][1][0] for i in range(len(self.ratingMatrix))]
        # if rating_mode == "scroll":
        #     self.scores = [self.ratingMatrix[i][1][1] for i in range(len(self.ratingMatrix))]
        # if rating_mode == "clicks":
        #     self.scores = [self.ratingMatrix[i][1][2] for i in range(len(self.ratingMatrix))]
        # if rating_mode == "merge":
        #     pass # Juntar os ratings 

        # self.x = list(zip(self.data, self.scores))

    def load_byte_file(self, path):
            with open(path, 'rb') as f:
                obj = pkl.load(f)
            return obj


    def load(self):

        self.user2id = self.load_byte_file(f"{self.source_path}/train/User2Index.pkl")
        self.item2id = self.load_byte_file(f"{self.source_path}/train/Item2Index.pkl")
        self.userIter = self.load_byte_file(f"{self.source_path}/train/UserItemIteractions{self.data_mode}.pkl")
        self.itemDoc = self.load_byte_file(f"{self.source_path}/train/Item2Document.pkl")
        self.userDoc = self.load_byte_file(f"{self.source_path}/train/User2Document{self.user_doc_mode}.pkl")
        # self.ratingMatrix = load_byte_file(f"{self.source_path}/{mode}/RatingMatrix.pkl")        


    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
