import os
import random
import math
import matplotlib.pyplot as plt

os.chdir('/Users/FanFan/trec07p/full')
ResultsList=[]
FileNames=[]
#read a file, each line of this file contains 2.email name and 1.this email belongs to spam or ham
with open("./index","r") as f1:
    for i in f1:
        arr = i.strip().split()
        ResultsList.append(arr[0])
        FileNames.append(arr[1])

train_ham=[]  #ham emails list
train_spam=[] #spam emails list
len_FileNames=len(FileNames)
actual_results=[]  #list of results as spam or ham emails for test set, spam=1, ham=0
test_files=[] #test file names list
train_index=[] #random picks around 75% emails as train set
test_index=[] #remaining around 25% as test set

test_percentage=0.75
for i in range(len(FileNames)):
    if random.random()<test_percentage:
        train_index.append(i)
        if ResultsList[i]=="ham":
            train_ham.append(FileNames[i])
        else:
            train_spam.append(FileNames[i])
    else:
        test_index.append(i)
        if ResultsList[i]=="ham":
            actual_results.append(0)
        if ResultsList[i]=="spam":
            actual_results.append(1)
        test_files.append(FileNames[i])

train_count=len(train_index)
test_count=len(test_index)

train_ham_words={}  #words that appear in train set ham emails
train_spam_words={}  #words that appear in test set spam emails

#read words and build dictionary from spam/ham emails of training set
def word_dic(emailList,wordsInEmail):  
    for i in range(len(emailList)):
        if i%100==0:
            print i
        emailList_set=set()
        with open (emailList[i],"r") as f:
            for j in f:
                arr=j.strip().split()
                for n in range(len(arr)):
                    emailList_set.add(arr[n])
        for word in emailList_set:
            if word in wordsInEmail:
                wordsInEmail[word]=wordsInEmail[word]+1
            else:
                wordsInEmail[word]=1

word_dic(train_ham,train_ham_words)
word_dic(train_spam,train_spam_words)
len_train_spam=len(train_spam)
len_train_ham=len(train_ham)
pre_spam=1.0*len_train_spam/train_count #prior probability of spam emails
pre_ham=1.0*len_train_ham/train_count  #prior probability of ham emails

#conditional probability of ham emails
for word in train_ham_words:
    train_ham_words[word]=1.0*(train_ham_words[word]+1)/(len_train_ham+1)

#conditional probability of spam emails
for word in train_spam_words:
    train_spam_words[word]=1.0*(train_spam_words[word]+1)/(len_train_spam+1)

log_prior_spam = math.log(pre_spam) #log of prior spam probability
log_prior_ham =math.log(pre_ham)  #log of prior ham probability
log_spam_not_shown=math.log(1.0/(len_train_spam+1)) #laplace smoothing for not shown words in train spam sets
log_ham_not_shown=math.log(1.0/(len_train_ham+1))  #laplace smoothing for not shown words in train ham sets
log_results=[]
actual_spam_count=0
actual_ham_count=0

for i in range(len(test_files)):
    if actual_results[i]==1:
        actual_spam_count=actual_spam_count+1
    else:
        actual_ham_count=actual_ham_count+1
    if i%100 == 0:
        print i

    test_set=set()
    log_condi_spam=0.0 #log of conditional probability for spam
    log_condi_ham=0.0  #log of conditional probability for ham

    with open(test_files[i],"r") as f:
        for j in f:
            arr=j.strip().split()
            for n in range(len(arr)):
                test_set.add(arr[n])
    for word in test_set:
        if word not in train_spam_words:
            log_condi_spam=log_condi_spam+log_spam_not_shown
        else:
            log_condi_spam=log_condi_spam+math.log(train_spam_words[word])

        if word not in train_ham_words:
            log_condi_ham=log_condi_ham+log_ham_not_shown
        else:
            log_condi_ham=log_condi_ham+math.log(train_ham_words[word])

    logfinal=log_condi_spam+log_prior_spam-log_condi_ham-log_prior_ham
    log_results.append(logfinal)

#different threshold values between min and max log_results
min_loglist=int(min(log_results))+1
max_loglist=int(max(log_results))-1
points=100
interval_loglist=int(1.0*(max_loglist-min_loglist)/points)
TP_FP_list=[]

#for different threshold, get true positive rate and  false positive rate for each threshold
for i in range(min_loglist,max_loglist,interval_loglist):
    TP_temp=0
    TN_temp=0
    for j in range(len(log_results)):
        if (log_results[j]>i) and (actual_results[j]==1):
            TP_temp=TP_temp+1
        if (log_results[j]<i) and (actual_results[j]==0):
            TN_temp=TN_temp+1
    TP_FP_list.append((1.0*TP_temp/actual_spam_count,(1-1.0*TN_temp/actual_ham_count)))

TP_FP_list=sorted(TP_FP_list, key=lambda x:x[1],reverse=False)

#calculate area under curve
AUC=0
x_old=0
for i in range(1,len(TP_FP_list)):

    if TP_FP_list[i][1]!=x_old:
        AUC=AUC+(TP_FP_list[i][1]-x_old)*(TP_FP_list[i][0]+TP_FP_list[i-1][0])/2
        x_old=TP_FP_list[i][1]

print 'auc:', AUC
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot([t[1] for t in TP_FP_list], [t[0] for t in TP_FP_list], color='darkorange',
         lw=lw, label='ROC curve (area = %0.9f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()







