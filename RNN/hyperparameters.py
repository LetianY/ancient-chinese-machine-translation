class Hyperparams:
    """
    Hyperparameters
    data files:
        m_e for english translation task (Pre-qin + ZiZhiTongJian)
        c_m for mandarin translation task (24 History)
    """

    # data file
    source_data_m_e =  "data/train_Pre-Qin+ZiZhiTongJian_m_utf8.txt"
    target_data_m_e = "data/train_Pre-Qin+ZiZhiTongJian_e_utf8.txt"
    source_data_c_m = "data/train_24-histories_c_utf8.txt"
    target_data_c_m = "data/train_24_histories_m_utf8.txt"

    # splitted data file
    source_train_m_e = "data_splited/pre_qin_train_m.txt"
    target_train_m_e = "data_splited/pre_qin_train_e.txt"
    source_test_m_e = "data_splited/pre_qin_test_m.txt"
    target_test_m_e = "data_splited/pre_qin_test_e.txt"

    source_train_c_m = "data_splited/24_history_train_c.txt"
    target_train_c_m = "data_splited/24_history_train_m.txt"
    source_test_c_m = "data_splited/24_history_test_c.txt"
    target_test_c_m = "data_splited/24_history_test_m.txt"

    # # training
    embed_size = 256
    hidden_size = 512
    batch_size = 8  # alias = N
    batch_size_valid = 32
    lr = (
        0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    )
    logdir = "logdir"  # log directory

    model_dir = "./models/"  # saving directory

    # # model
    maxlen = 50  # Maximum number of words in a sentence. alias = T.
    min_cnt = 0  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 25
    num_heads = 8
    dropout_rate = 0.4
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 1  # epoch of model for eval
    eval_script = 'scripts/validate.sh'
    check_frequency = 5  # checkpoint frequency
