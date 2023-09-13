general_config = {
    # Definir tamanho do vocab dinamicamente
    'vocab_size': 768,
    'word_dim': 1,

    # NN config:
    'filters_num': 100,
    'kernel_size': 3,
    'fc_dim': 32,
    'drop_out': 0.5,
    'num_fea': 1,

    # Dataset Configuration
    'user_num': 100 + 2,
    'item_num': 100 + 2,

    # SelfAttention Config
    'num_heads': 8,
    'id_emb_size': 32,

    # Predict Method Config
    'output': 'nfm',
    'ui_merge': 'cat',
    'r_id_merge': 'cat',
}

train_test_config = {
    "data_mode": "Val",
    "user_doc_mode": "Mean"
}