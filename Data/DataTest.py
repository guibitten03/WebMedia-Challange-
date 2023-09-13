import pickle as pkl

class TestData():
    def __init__(self, root_path, raw_path, user_doc_mode="Mean"):

        self.root_path = root_path
        self.raw_path = raw_path
        self.user_doc_mode = user_doc_mode
        
        print("Loading Files...")
        self.load()

        print("Extracting Features...")
        self.extract_features()


    def load_byte_file(self, path):
        with open(path, 'rb') as f:
            obj = pkl.load(f)
        return obj


    def load(self):
        self.val_data = self.load_byte_file(f"{self.raw_path}/df_validacao_candidate.pkl")
        self.userDoc = self.load_byte_file(f"{self.root_path}/User2Document{self.user_doc_mode}.pkl")
        self.itemDoc = self.load_byte_file(f"{self.root_path}/Item2Document.pkl")
        self.index2item = self.load_byte_file(f"{self.raw_path}/index2item.pkl")
        self.index2user = self.load_byte_file(f"{self.raw_path}/index2user.pkl")


    def extract_features(self):
        self.users = list(self.val_data['user'].values)
        # self.itens = list(self.index2item.keys())
        # val_itens = list(self.val_data['item_validacao'].values)
        cand_itens = list(self.val_data['item_candidate'].values)

        self.itens = cand_itens
        # self.itens = [i_0 + i_1 for (i_0, i_1) in zip(val_itens, cand_itens)]
        

        # O que vou precisar para montar essa nova estapa de teste? 
        '''
            Preciso pegar todos os usuários da validação e os documentos deles.
            Preciso pegar todos os itens da base.
            Preciso calcular a similaridade entre um usuário para todos os itens.
        '''

        # for (user, itens_val, itens_recen) in zip(self.val_data['user'], )