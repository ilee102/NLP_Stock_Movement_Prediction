import collections, random
import string, math, operator
import glob, os, re, matplotlib
import datetime
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
months = ['January','February','March','April','May','June', \
          'July','August','September','October','November','December']

nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()

# given two defaultdicts, this function outputs its dot product.
def dotProduct(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))
            
def increment(d1, scale, d2):
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale   

# this function extracts stock data from file.
def extractStockData(filename):
    map = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if line[0].isdigit():
                map[line.split(',')[0]] = line.split(',')[1:]
            line = f.readline()
    return map

# this function converts the time to string to be used for further formatting, 
# with n representing the number of days from input "time".
def convTimetoString(time,n):   
    split = time.split(',')
    todayDate, todayMonth, todayYear = int(re.sub('\D','',split[1].strip().split()[1])), \
                    months.index(split[1].strip().split()[0])+1, \
                    int(split[2].strip()[:4])
    
    today = datetime.datetime(todayYear,todayMonth,todayDate)
    yday, tmrw = today, today
    m = n
    while n > 0:
        
        yday = yday - datetime.timedelta(1) 
        
        if yday.weekday() < 5:
            n -= 1
     
            
    while m > 0:
        tmrw = tmrw + datetime.timedelta(1) 
        
        if tmrw.weekday() < 5:
            m -= 1

    ydayDate = str(yday.year) + '-' + yday.strftime('%m') + '-' + yday.strftime('%d')
    tmrwDate = str(tmrw.year) + '-' + tmrw.strftime('%m') + '-' + tmrw.strftime('%d')
            
    return ydayDate,tmrwDate

# this function extracts the label given the time ranges by using the stock data map extracted
# from file.
def extractY(map,date_previous,date_end):
    if date_previous not in map:
        split = date_previous.split('-')
        y,m,d = int(split[0]), int(split[1]), int(split[2])
        conv = datetime.datetime(y,m,d)
        while date_previous not in map:
            conv = conv - datetime.timedelta(1)
            date_previous = str(conv.year) + '-' + conv.strftime('%m') + '-' + conv.strftime('%d')
    if date_end not in map:
        split = date_previous.split('-')
        y,m,d = int(split[0]), int(split[1]), int(split[2])
        conv = datetime.datetime(y,m,d)
        while date_end not in map:
            conv = conv + datetime.timedelta(1)
            date_end = str(conv.year) + '-' + conv.strftime('%m') + '-' + conv.strftime('%d')
    closingOrig = float(map[date_previous][3])
    closingNew = float(map[date_end][3])
    if closingNew - closingOrig > 0:
        return 1
    else:
        return -1

# this function extracts unigram freature from a given earnings transcript.    
def extractWordFeatures(filename):
    map = collections.defaultdict(int)
    with open(filename,'r', encoding='ISO-8859-1') as f:
        line = f.readline()
        cnt = 1
        dayInd = 1000000
        while line:
            if line.split(',')[0] in days:
                time = line
                dayInd = cnt
            if cnt > dayInd:
                treated = line.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
                for i in treated.split():
                    key = lemmatizer.lemmatize(i.lower())
                    if len(key) > 2 and not key.isdigit() and key not in stop_words:
                       map[key] += 1
            cnt += 1
            line = f.readline()  
    
    return time, map

# this function extracts n-gram features from a given earnings transcript.
def extractNgramFeatures(filename,n):
    map = collections.defaultdict(int)
    with open(filename,'r', encoding='ISO-8859-1') as f:
        line = f.readline()
        cnt = 1
        dayInd = 1000000
        while line:
            if line.split(',')[0] in days:
                time = line
                dayInd = cnt
            if cnt > dayInd:
                treated = line.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
                split = [w for w in treated.lower().split() if w not in stop_words]
                for i in range(len(split)):
                    key = tuple()
                    for j in range(n):
                        key +=  tuple(lemmatizer.lemmatize(split[(i+j) % len(split)]))
                    map[key] += 1
            cnt += 1
            line = f.readline()

  
    
    return time, map

# this function learns the weights in the linear classification model by minimizing
# hinge-loss via stochastic gradient descent.
def learnPredictor(trainExamples, testExamples, numIters, eta):
    weights = {}  
    train,test = [],[]
    for k in range(numIters):
        #print(len(list(weights.items())))
        #curr = copy.deepcopy(weights)
        for tuple in trainExamples:
            if tuple[1]*dotProduct(weights,tuple[0]) < 1:
                increment(weights,tuple[1]*eta,tuple[0])
        trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(x, weights) >= 0 else -1))
        devError = evaluatePredictor(testExamples, lambda x : (1 if dotProduct(x, weights) >= 0 else -1))
        #print(("Official: train error = %s, dev error = %s" % (trainError, devError)))
        train.append(trainError)
        test.append(devError)

        
    return weights, train , test

# this function evaluates the percent error of the prediction.
def evaluatePredictor(examples, predictor):
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

# this function is to see the values of the weights of the model.
def seeWeights(flag, n, x):
    x_sorted = sorted(x.items(), key=operator.itemgetter(1))
    if flag == 'bottom':
        print(x_sorted[:n])
    elif flag == 'top':
        print(x_sorted[len(x)-n:])

#this function extracts the samples and labels from a given company earnings call directory
#using unigram model.
def extractAll(company,totalSetPos,totalSetNeg,n):
    filename = company+'/MacroTrends_Data_Download_'+company+'.csv'
    stockData = extractStockData(filename)
    filedir = company
    count = 0
    for p in glob.glob(os.path.join(filedir,'FY*')):
        for pathname in glob.glob(os.path.join(p,'*.txt')): 
            time, features = extractWordFeatures(pathname)
            #time, features = extractBigramFeatures(pathname)
            #print(time, convTimetoString(time,n))
            Y = extractY(stockData,convTimetoString(time,n)[0],convTimetoString(time,n)[1])
            count += 1
            #print('y: ',Y)
            if Y == 1:
                totalSetPos.append((features,Y))
            else:
                totalSetNeg.append((features,Y))
    print('Total count from ',company,': ',count)

#this function extracts the samples and labels from a given company earnings call directory
#using n-gram model.
def extractAllNGram(company,totalSetPos,totalSetNeg,n,ngram):
    filename = company+'/MacroTrends_Data_Download_'+company+'.csv'
    stockData = extractStockData(filename)
    filedir = company
    count = 0
    for p in glob.glob(os.path.join(filedir,'FY*')):
        for pathname in glob.glob(os.path.join(p,'*.txt')): 
            time, features = extractNgramFeatures(pathname,ngram)
            #time, features = extractBigramFeatures(pathname)
            #print(time, convTimetoString(time,n))
            Y = extractY(stockData,convTimetoString(time,n)[0],convTimetoString(time,n)[1])
            count += 1
            #print('y: ',Y)
            if Y == 1:
                totalSetPos.append((features,Y))
            else:
                totalSetNeg.append((features,Y))
    print('Total count from ',company,': ',count)

# the following functions are used to do the entire process of extracting data,
# setting the train-test split, and running the linear classification models.
def oneRun(nIter,eta,totalSetPos,totalSetNeg): 
    m,n = len(totalSetPos),len(totalSetNeg)
    #print("size of +1 set: {}".format(m))
    #print("size of -1 set: {}".format(n))
    random.shuffle(totalSetPos)
    random.shuffle(totalSetNeg)
    trainSet = totalSetPos[:int(0.6*m)] + totalSetNeg[:int(0.6*n)]
    testSet = totalSetPos[int(0.6*m):] + totalSetNeg[int(0.6*n):]
    random.shuffle(trainSet)
    random.shuffle(testSet)
    
    weights, train, test = learnPredictor(trainSet, \
                                      testSet, nIter, eta)
    
    return train,test

# the following functions are used to do the entire process of extracting data,
# setting the train-test split, and running the linear classification models.
def oneMegaRun(ndays,eta):
    tPos, tNeg = [], [] 
    
    extractAll('MSFT',tPos,tNeg,ndays)
    extractAll('V',tPos,tNeg,ndays)
    
    train,test = oneRun(60,eta,tPos,tNeg) 
    train,test = np.array(train), np.array(test)
    for i in range(10):
        print(i)
        r,e = oneRun(60,eta,tPos,tNeg)
        train += r
        test += e
    
    return train/21,test/21
# the following functions are used to do the entire process of extracting data,
# setting the train-test split, and running the linear classification models.
def oneMegaRunNGram(n,eta):
    tPos, tNeg = [], [] 
    ndays = 5
    extractAllNGram('MSFT',tPos,tNeg,ndays,n)
    extractAllNGram('V',tPos,tNeg,ndays,n)
    
    train,test = oneRun(60,eta,tPos,tNeg) 
    train,test = np.array(train), np.array(test)
    for i in range(10):
        print(i)
        r,e = oneRun(60,eta,tPos,tNeg)
        train += r
        test += e
    
    return train/21,test/21

# this function is used to take all the samples and create a set with all the features.
def extractTotalFeatureTemplate(samples):
    result = set()
    for i in range(len(samples)):
        for feature in samples[i]:
            result.add(feature)
    return result

# this function is to turn the samples into an n by m list of numbers, where n represents
# sample number, and m represents the total feature size.  Used for neural network inputting.
def encode(template, samples):
    temp = list(template)
    result = [[0 for _ in range(len(temp))] for _ in range(len(samples))]
    
    for i in range(len(samples)):
        for feature in samples[i]:
            result[i][temp.index(feature)] = samples[i][feature]    

    return result

#this function runs the MLP classification model.
def neuralRun(neurons,ndays,l):
    
    totalSetPos,totalSetNeg = [], [] 
    
    #extractAll('MSFT',totalSetPos,totalSetNeg,ndays)
    #extractAll('V',totalSetPos,totalSetNeg,ndays)
    
    extractAllNGram('MSFT',totalSetPos,totalSetNeg,ndays,l)
    extractAllNGram('V',totalSetPos,totalSetNeg,ndays,l)
    
    
    m,n = len(totalSetPos),len(totalSetNeg)
    
  
    random.shuffle(totalSetPos)
    random.shuffle(totalSetNeg)
    fraction = 0.6
    trainSet = totalSetPos[:int(fraction*m)] + totalSetNeg[:int(fraction*n)]
    testSet = totalSetPos[int(fraction*m):] + totalSetNeg[int(fraction*n):]
    random.shuffle(trainSet)
    random.shuffle(testSet)
    
    
    X_train, Y_train = [trainSet[i][0] for i in range(len(trainSet))], \
        [trainSet[i][1] for i in range(len(trainSet))]
        
    X_test, Y_test = [trainSet[i][0] for i in range(len(testSet))], \
        [trainSet[i][1] for i in range(len(testSet))]
        
    set_features = extractTotalFeatureTemplate(X_train + X_test)
    print('number of features: ',len(set_features))
    
    encoded_train = encode(set_features,X_train)
    encoded_test = encode(set_features, X_test)
        
    model = MLPClassifier(hidden_layer_sizes=(neurons,), max_iter = 500, activation='relu', early_stopping=False)
    model.fit(encoded_train, Y_train)
    
    Y_train_pred = model.predict(encoded_train)
    Y_test_pred = model.predict(encoded_test)

    return accuracy_score(Y_train, Y_train_pred), accuracy_score(Y_test, Y_test_pred)


def main():   


# below is for collecting data with neural network model. 
    
    train_full, test_full = [], []
    train_full1, test_full1 = [], []
    train_full2, test_full2 = [], []
    for i in range(1,20):
        print(i)
        train1, train2, train3, test1, test2, test3 = 0,0,0,0,0,0
        for _ in range(20):
            t1, s1 = neuralRun(i,30,1)
            t2, s2 = neuralRun(i,30,1)
            t3, s3 = neuralRun(i,30,1)
        
            train1 += t1
            train2 += t2
            train3 += t3
            test1 += t1
            test2 += t2
            test3 += t3
        
        
        train_full.append(train1/20)
        train_full1.append(train2/20)
        train_full2.append(train3/20)
        
        test_full.append(test1/20)
        test_full1.append(test2/20)
        test_full2.append(test3/20)

    matplotlib.pyplot.plot(train_full,'k*',test_full,'ko')
    matplotlib.pyplot.plot(train_full1,'b*',test_full1,'bo')
    matplotlib.pyplot.plot(train_full2,'g*',test_full2,'go')
   # matplotlib.pyplot.plot(train4,'r-',test4,'r--')
    matplotlib.pyplot.legend(['train: 1', 'test: 1','train: 2', 'test: 2','train: 5','test: 5'], \
                             loc='center left', bbox_to_anchor=(1, 0.5))
    matplotlib.pyplot.xlabel('# of Neurons')
    matplotlib.pyplot.ylabel('% Accuracy')    
    
  
#below is for collecting data with linear classification model.
    
    train1, test1 = oneMegaRunNGram(1,0.1)
    train2, test2 = oneMegaRunNGram(2,0.1)
    train3,test3 = oneMegaRunNGram(3,0.1)
    train4,test4 = oneMegaRunNGram(5,0.1)

    matplotlib.pyplot.plot(train1,'k-',test1,'k--')
    matplotlib.pyplot.plot(train2,'b-',test2,'b--')
    matplotlib.pyplot.plot(train3,'g-',test3,'g--')
    matplotlib.pyplot.plot(train4,'r-',test4,'r--')
    matplotlib.pyplot.legend(['train: 1', 'test: 1','train: 2', 'test: 2','train: 3','test: 3', 
                              'train: 5','test: 5'], \
                             loc='center left', bbox_to_anchor=(1, 0.5))
    matplotlib.pyplot.xlabel('Iterations')
    matplotlib.pyplot.ylabel('% Error')


if __name__ == '__main__':
    main()
