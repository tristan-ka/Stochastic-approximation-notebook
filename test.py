import pandas as pd
import os
import re

from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer("english")

from itertools import groupby
import nltk
stopwords = nltk.corpus.stopwords.words('english')

# ---------------------------------------------------------------------------------------------
# ----------------------------------- TOKENIZE WITH MAPPING -----------------------------------
# ---------------------------------------------------------------------------------------------
 
def compute_spans(sent):
    '''Compute the span of each word in the sentence sent
    
    :param str sent: Sentence from which we want to compute the spans of its words.
    :return: a list of spans and a list of words.
    '''

    sent = sent + ' ' 
    space = re.compile(' ')
    split_span = re.finditer(space, sent)
    words = sent.split()
    spans = [(m.start(0)-len(word), m.start(0)) for m, word in zip(split_span,words)]
    return spans, words

def sub_map(words, old_parents, regex, sub_str):
    '''
    subtitution function - replace the regex in compiled with sub_str.

    :param list(str) words: list of words corresponding to the sentence.
    :param list(int) old_parents: list of indices for each word.
    :param regex: re.compiled corresponding to the regular expression to find in sent.
    :param str sub_str: string for the replacement.
    :return: a list of processed words and a list of index corresponding to their indices in the input sentence (raw sentence).
    '''
    new_words = []
    new_parents = []
    
    for word, parent in zip(words, old_parents):
        lw = re.sub(regex, sub_str, word).split(' ')
        new_words.extend(lw)
        new_parents.extend([parent for _ in range(len(lw))])
        
    return new_words, new_parents

def filter_words(words, indices, keep_stopwords):
    '''Filter words to remove all non a-Z characters.
    If option keep_stopwords is set to false then all the stopwords define in nltk english stop words will be removed
    
    :param list(str) words: list of words to filter
    :param list(idx) indices: list of index corresponding to the position of the word in the raw sentence
    :param bool keep_stopwords: Boolean to set wether to keep or remove stopwords.
    :return: filtered list of indices and filtered list of words
    :rtype: list, list
    '''

    filtered_words = []
    filtered_indices = []
    if keep_stopwords:
        for word, index in zip(words, indices):
            if (re.search('[a-zA-Z]', word)):
                filtered_words.append(word)
                filtered_indices.append(index)
    else:
        for word, index in zip(words, indices):
            if (re.search('[a-zA-Z]', word)) and (word not in stopwords):
                filtered_words.append(word)
                filtered_indices.append(index)
    return filtered_words, filtered_indices


def tokenize_map(input_sent, keep_stopwords = True, stemming = True, lower = True):
    '''Tokenize a sentence and save the position of each words so that the raw text can be extracted from it.
    
    :param str input_sent: input sentence to tokenize
    :param keep_stopwords: boolean to set wether to keep or remove stopwords, defaults to True
    :param stemming: boolean to set wether to stem words or not, defaults to True
    :param lower: boolean to set wether to lower words or not, defaults to True
    :return: list of processed words
    '''

    #Listing the number of transformation to apply to the sentence
    url = re.compile('https?[^ \t\n\r\f\v]*|http')
    junk = re.compile('["¤#&()*+-/<=>@[\]^_`{|}~\\\(\)\'\"\*\`\´\‘\’…\\\/\{\}\|\+><~\[\]\“\”%=\$§\.;:]')
    percent = re.compile('%')
    dash = re.compile('-')
    number = re.compile(r'\d+(?:,|.\d+)+(?:,|.\d+)?')
    punct = re.compile('[.!?,:;]')

    def subfct1(matchobj):         
        return ' ' + matchobj.group(0) + ' '
    # number = re.compile('(^[0-9]+)|([0-9]+)')

    #Listing the associated replacement string in the sentence
    url_str = ' url '
    percent_str = ' percent '
    number_str = ' number '
    dash_str = ''
    junk_str = ' '

    regex_list = [url, percent, dash, number, junk]
    str_list = [url_str, percent_str, dash_str, number_str, junk_str]

    words = input_sent.split(' ')
    indices = list(range(0,len(words)))
    # Executing all the transformation
    for regex, sub_str in zip(regex_list, str_list):
        words, indices = sub_map(words, indices, regex, sub_str)
        
    words, indices = filter_words(words, indices, keep_stopwords)

    if lower:
        words = [t.lower() for t in words]
       
    if stemming:
        #stemming words
        words = [stemmer.stem(t) for t in words] 
    
    return words, indices

def recover_raw_text(indices_to_show, input_sent):
    spans, words = compute_spans(input_sent)
    indices = list(range(0,len(words)))
    span_map = dict(zip(indices, spans))

    span_list = []
    for index in indices_to_show:
        span_list.append(span_map[index])
        
    first_span = span_list[0]
    last_span = span_list[-1]   
    return input_sent[first_span[0]:last_span[1]]

# ---------------------------------------------------------------------------------------------
# --------------------------------- TOKENIZE WITHOUT MAPPING ----------------------------------
# ---------------------------------------------------------------------------------------------

def tokenize(line, stem=True, lower=True, keep_stopwords=False, joined=True):
    '''Tokenize the sentence (str) "line" with desired options
    
    :param str line: sentence to tokenize.
    :return: A list of tokens if joined == False or A string of joinned words if joined == True
    '''

    url = re.compile('https?[^ \t\n\r\f\v]*|http')
    junk = re.compile('["¤#&()*+-/<=>@[\]^_`{|}~\\;\(\)\'\"\*\`\´\‘\’…\\\/\{\}\|\+><~\[\]\“\”%=\$§]')
    percent = re.compile('%')
    number = re.compile('[0-9]+(\.[0-9][0-9]?)?')
    punct = re.compile('[.!?,:;]')
    rep = re.compile(r'(.)\1{2,}')
    
    if line.startswith('"') : 
        line = line[1:]     
    if line.endswith('"') :        
        line = line[:-1]            
            
    line = re.sub(url,' url ', line) # replace every url with ' url '
    line = re.sub(percent, ' percent ', line)         
    ### Important replace all non unicode characters!     
    line = re.sub(r'[^\x00-\x7F]+',' ', line)             
    
    def subfct1(matchobj):         
        return ' ' + matchobj.group(0) + ' '  
    
    line = re.sub(punct,subfct1, line) # separate the punctuation from the words    
  
    def subfct2(matchobj):         
        return matchobj.group(0)[:2]     
    
    line = re.sub(rep, subfct2,line) # keep maximum 2 consecutive identical character         
    line = re.sub(junk,' ', line) #throw away junk character
   
    line = re.sub(number,' number ',line) # replace every number with ' number '   
    
    splitted = line.split()  
    
    L = []     
    for word in splitted :       
        if lower:         
            word = word.lower()     
        else:     
            pass      
        L.extend(word.split())       
        
    line = ' '.join(L)   
    line = ' '.join([k for k,v in groupby(line.split())]) #suprr repeated word:    
    line = [t for t in line.split()]  
    
    filtered_tokens = []   
    for token in line:   
        if keep_stopwords:      
            if (re.search('[a-zA-Z]', token)):             
                filtered_tokens.append(token)       
        else:           
            if (re.search('[a-zA-Z]', token)) and (token not in stopwords):     
                filtered_tokens.append(token)   
                    
    if stem==True:    
        stems = [stemmer.stem(t) for t in filtered_tokens]    
    else:       
        stems = [t for t in filtered_tokens]          
    
    if joined == True:
        stems = " ".join(stems)
    
    return stems
                                
