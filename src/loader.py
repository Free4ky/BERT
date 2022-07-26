# get textst from VK by vk api
import matplotlib.pyplot as plt
import pandas as pd
import vk_requests
import time
import random
import seaborn as sns


class MyLoader():

    def __init__(self):
        self.numIterations = 100
        token = '57506d6430f9b05657f724126d7eb556c63d11b8921b1ef88dac93236e9cf58361ffa5fff65e822669001'
        self.api = vk_requests.create_api(service_token=token)
        self.classes = {
            'educational': [
                'mirea_official',
                'hse.lectorium',
                'msucareer',
                'bmstu1830',
                'miptfpmi',
                'finuniver_kf',
                'nust_misis',
                'gubkin.university',
                'mephi_official',
                'kgu_kaluga',
                'spb1724',
                'spbsetu'],
            'other': [
                'kylinar_vk',
                'kinopoisk',
                'anecdotes',
                'overhear',
                'palatka_tru',
                'ribalka2017',
                'airsoft_rus',
                'club13704425',
                'f.recept',
                'klgshest',
                'lentach']
        }

    def loadDf(self, path):
        df = pd.read_csv(path)
        df = dropEmptyStrings(df)
        return df

    def getDf(self):
        delay = [0.34, 0.44, 0.4, 0.43, 0.5]
        step = 100
        self.educational = []
        self.other = []
        counter = 1
        # get walls info
        for key, value in self.classes.items():
            shift = 0
            buffer = []
            counter = 1
            for domain in value:
                print(f'{key}: {counter}/{len(value)}')
                counter += 1
                for i in range(self.numIterations):
                    temp = self.api.wall.get(
                        domain=domain, count=step, offset=shift)
                    time.sleep(random.choice(delay))
                    shift += step
                    buffer.extend(temp['items'])
            if key == 'educational':
                self.educational.append(buffer)
            elif key == 'other':
                self.other.append(buffer)

        educationalText = []
        textLen = []
        for post in self.educational[0]:
            educationalText.append(post['text'])
            textLen.append(len(post['text']))

        otherText = []
        for post in self.other[0]:
            otherText.append(post['text'])
            textLen.append(len(post['text']))

        labelsEdu = ['educational' for i in range(len(educationalText))]
        labelsOther = ['other' for i in range(len(otherText))]

        # create dataframe
        d = {'category': labelsEdu+labelsOther,
             'length': textLen, 'text': educationalText+otherText}
        df = pd.DataFrame(data=d)
        df = dropEmptyStrings(df)
        return df


def dropEmptyStrings(df):
    df = df[df['length'] > 0]
    return df


def showGist(df):
    # plt.hist(df['length'], color = 'blue', edgecolor = 'black', bins = int(180/10))
    # plt.title('Распределение длины постов')
    # #plt.xlim(1000)
    # plt.show()
    sns.histplot(data=df, x='length', hue='category')
    plt.title('Распределение длины постов')
    plt.xlim(0, 5000)
    plt.show()
