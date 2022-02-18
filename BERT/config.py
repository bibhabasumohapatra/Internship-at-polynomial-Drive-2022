class config():
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 2
    EPOCHS = 5
    BERT_PATH = '../input/bert-base-uncased'
    MODEL_PATH = 'model.bin'
    TRAINING_FILE = "../input/amazon-dataset-csv-generator/PolynomialInternshipDrive2022.csv"
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
