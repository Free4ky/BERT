from dataclass import splitDf
from loader import*
from model import*

if __name__ == '__main__':
    ml = MyLoader()
    df = ml.loadDf('data\\vkgroups.csv')
    df_train, df_val, df_test = splitDf(df)
    print(len(df_train), len(df_val), len(df_test))

    print(torch.cuda.get_device_name(0))

    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6

    model.train(model, df_train, df_val, LR, EPOCHS)