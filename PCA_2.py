# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:35:14 2024

@author: Omargade
"""
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


transactions=[
    ['milk','bread','butter'],
    ['bresd','eggs'],
    ['milk','bresd','eggs','butter'],
    ['bread','eggs','butter'],
    ['milk','bread','butter']
]


#step1:- convert the dataset into the format suitable for Apriori using 

te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df=pd.DataFrame(te_ary,columns=te.columns_)

#step2:apply the apriori algio to find the frequent itemset
frequent_itemsets=apriori(df,min_support=0.5,use_colnames=True)

#step3:- generate association rules from the frequent itemsets
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)


#step4:- output the result
print("Frequent itemset:-")
print(frequent_itemsets)

print("\n association rules:-")
print(rules[['antecedents','consequents','support','confidence','lift']])

#######################################################################

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


#step1:- simulating healthcare transactions(symptons/diseases/treatment)
healthcare_data=[
    ['fever','cough','COVID-19'],
    ['cough','sore throat','flu'],
    ['fever','cough','shortness of breath','COVID-19'],
    ['cough','sore throat','flu','headache'],
    ['fever','body ache','flu'],
    ['fever','cough','COVID-19','shortness of breath'],
    ['sore throat','headache','cough'],
    ['body ache','fatigue','flu']
]

te=TransactionEncoder()
te_ary=te.fit(healthcare_data).transform(healthcare_data)
df=pd.DataFrame(te_ary,columns=te.columns_)

#step2:apply the apriori algio to find the frequent itemset
frequent_itemsets=apriori(df,min_support=0.3,use_colnames=True)


#step3:- generate association rules from the frequent itemsets
rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)


#step4:- output the result
print("Frequent itemset:-")
print(frequent_itemsets)


print("\n association rules:-")
print(rules[['antecedents','consequents','support','confidence','lift']])

############################################################################

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


#step1:- simulating healthcare transactions(symptons/diseases/treatment)
transactions=[
    ['laptop','mouse','keyboard'],
    ['smartphones','headphones'],
    ['laptop','mouse','headphones'],
    ['smartphone','charger','phone case'],
    ['laptop','mouse','moniter'],
    ['headphones','smartwatch'],
    ['laptop','keyboard','moniter'],
    ['smaertphone','charger','phone case','screen protector'],
    ['mouse','keyboard','moniter'],
    ['smartphone','headphone','smartwatch']
    
]

te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)
df=pd.DataFrame(te_ary,columns=te.columns_)

#step2:apply the apriori algio to find the frequent itemset
frequent_itemsets=apriori(df,min_support=0.2,use_colnames=True)


#step3:- generate association rules from the frequent itemsets
rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.5)


#step4:- output the result
print("Frequent itemset:-")
print(frequent_itemsets)


print("\n association rules:-")
print(rules[['antecedents','consequents','support','confidence','lift']])












