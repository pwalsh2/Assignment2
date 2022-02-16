import math
import string

eps = 0.0001

def distance_one_edits(w):
    # Return the set of strings that can be formed by applying one
    # delete/insert/substitution/transposition operation to word w

    # helper function for thetransposition function
    def swap(c, i, j):
        c = list(c)
        c[i], c[j] = c[j], c[i]
        return ''.join(c)
    #get alphabet list
    alphabet_string = string.ascii_lowercase 
    alphabet_list = list(alphabet_string)
   
    # deletion
    result = set()
    for i in range(len(w)):
        result.add(w[:i] + w[i+1:])

    # insertion
    for i in range(len(w)+1):
        for j in alphabet_list:
            result.add(w[:i]+ j + w[i:])

    # substitution
    for i in range(len(w)):
        for j in alphabet_list:
            result.add(w[:i] + j + w[i+1:])
      

    # transposition
    for i in range(len(w) -1):
        transposed = swap(w, i, i + 1)
        result.add(transposed.lower())

    return result




class UnigramLM:
    def __init__(self, fname):
        self.freqs = {}
        for line in open(fname):
            tokens = line.split()
            for t in tokens:
                
                self.freqs[t.lower()] = self.freqs.get(t.lower(), 0) + 1
        # Computing this sum once in the constructor, instead of every
        # time it's needed in
        # log_prob, speeds things up
        self.freqs["UNK"] = self.freqs.get("UNK", 0)
        self.num_tokens = sum(self.freqs.values())

    def log_prob(self, word):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        word = word.lower()
        if word in self.freqs:
            return math.log(self.freqs[word]+ 1) - math.log(self.num_tokens + len(self.freqs))
        else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            return math.log(1) - math.log(self.num_tokens + len(self.freqs))


    def prob(self, word):
        word = word.lower()
        if word in self.freqs:
            return (self.freqs[word] + 1) / (self.num_tokens + len(self.freqs))
        else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            return 1 / len(self.freqs)
   
    def in_vocab(self, word):
        return word in self.freqs

    def check_probs(self):
        # Hint: Writing code to check whether the probabilities you
        # have computed form a valid probability distribution is very
        # helpful, particularly when you start incorporating smoothing
        # (or interpolation). It can be a bit slow, however,
        # especially for bigram language models, so you might want to
        # turn these checks off once you're convinced things are
        # working correctly.

        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            assert 0 - eps < math.exp(self.log_prob(w)) < 1 + eps
        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
            sum([math.exp(self.log_prob(w)) for w in self.freqs]) < \
            1 + eps



class BigramLM:
    def __init__(self, fname):
        self.freqs = {}
        for line in open(fname):
            tokens = line.split()
            for t in range(len(tokens)-1):
                bi_gram_token = tokens[t].lower() + " " + tokens[t+1].lower()
                self.freqs[bi_gram_token] = self.freqs.get(bi_gram_token, 0) + 1
              
        self.uni_gram_lm = UnigramLM(fname)      
        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.num_tokens = sum(self.freqs.values())

    def log_prob(self, target_word, prior ):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        prior = prior.lower()
        target_word = target_word.lower()
        bigram = prior + " " + target_word
        if(self.uni_gram_lm.in_vocab(prior) and self.in_bigram_vocab(bigram)):
            return math.log(self.freqs[bigram] + 1) - math.log(self.uni_gram_lm.freqs[prior] + len(self.uni_gram_lm.freqs))
        elif(self.uni_gram_lm.in_vocab(prior)):
            return math.log(1) - math.log(self.uni_gram_lm.freqs[prior] + len(self.uni_gram_lm.freqs))
        else:
            return math.log(1) - math.log(len(self.uni_gram_lm.freqs))
       

    def log_probs_unsmoothed(self, target_word, prior):
         # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        prior = prior.lower()
        target_word = target_word.lower()
        bigram = prior + " " + target_word
        if(self.uni_gram_lm.in_vocab(prior) and self.in_bigram_vocab(bigram)):
            return self.freqs[bigram] / self.uni_gram_lm.freqs[prior]
        else:
            return 0


    def in_bigram_vocab(self, bigram):
        return bigram in self.freqs


    def in_vocab(self, word):
        return word.lower() in self.uni_gram_lm.freqs

    def check_probs(self):
        # Hint: Writing code to check whether the probabilities you
        # have computed form a valid probability distribution is very
        # helpful, particularly when you start incorporating smoothing
        # (or interpolation). It can be a bit slow, however,
        # especially for bigram language models, so you might want to
        # turn these checks off once you're convinced things are
        # working correctly.

        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            assert 0 - eps < math.exp(self.log_prob(w)) < 1 + eps
        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
            sum([math.exp(self.log_prob(w)) for w in self.freqs]) < \
            1 + eps



class InterpolatedLM:
    def __init__(self, fname, LambdaIn):
        
        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.biLM = BigramLM(fname)
        self.uniLM = UnigramLM(fname)
        self.Lambda = LambdaIn

    def log_prob(self, target_word, prior ):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        bi_log_prob = self.biLM.log_probs_unsmoothed(target_word, prior)
        uni_log_prob = self.uniLM.prob(target_word)

        return ((1-self.Lambda) * bi_log_prob + (self.Lambda)*uni_log_prob)

    def check_probs(self):
        # Hint: Writing code to check whether the probabilities you
        # have computed form a valid probability distribution is very
        # helpful, particularly when you start incorporating smoothing
        # (or interpolation). It can be a bit slow, however,
        # especially for bigram language models, so you might want to
        # turn these checks off once you're convinced things are
        # working correctly.

        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            assert 0 - eps < math.exp(self.log_prob(w)) < 1 + eps
        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
            sum([math.exp(self.log_prob(w)) for w in self.freqs]) < \
            1 + eps







if __name__ == '__main__':
    import sys

    # Look for the training corpus in the current directory
    train_corpus = 'corpus.txt' 

    # n will be '1', '2' or 'interp' (but this starter code ignores
    # this)
    n = sys.argv[1]

    # The collection of sentences to make predictions for
    predict_corpus = sys.argv[2]

    if(n == "1"):
        # Unigram has been chosen
        
        # Train the language model
        lm = UnigramLM(train_corpus)
    
        # lm_bi = UnsmoothedBrigramLM(train_corpus)
        # You can comment this out to run faster...
        # lm.check_probs()
    
        for line in open(predict_corpus):
        # Split the line on a tab; get the target word to correct and
        # the sentence it's in
            target_index,sentence = line.split('\t')
            target_index = int(target_index)
            sentence = sentence.split()
            target_word = sentence[target_index]

            # Get the in-vocabulary candidates (this starter code only
            # considers deletions)
            candidates = distance_one_edits(target_word)
            iv_candidates = [c for c in candidates if lm.in_vocab(c)]
            
            # Find the candidate correction with the highest probability;
            # if no candidate has non-zero probability, or there are no
            # candidates, give up and output the original target word as
            # the correction.
            best_prob = float("-inf")
            best_correction = target_word
            for ivc in iv_candidates:
                ivc_log_prob = lm.log_prob(ivc)
                if ivc_log_prob > best_prob:
                    best_prob = ivc_log_prob
                    best_correction = ivc

            print(best_correction)

    elif(n == "2"):
        # Bigram chosen
        # Train the language model
        lm = BigramLM(train_corpus)
    
        # lm_bi = UnsmoothedBrigramLM(train_corpus)
        # You can comment this out to run faster...
        # lm.check_probs()
    
        for line in open(predict_corpus):
            # Split the line on a tab; get the target word to correct and
            # the sentence it's in
            target_index,sentence = line.split('\t')
            target_index = int(target_index)
            sentence = sentence.split()
            target_word = sentence[target_index]
            previous_word = sentence[target_index - 1]
            next_word = sentence[target_index + 1]
            # Get the in-vocabulary candidates (this starter code only
            # considers deletions)
            candidates = distance_one_edits(target_word)
            iv_candidates = [c for c in candidates if lm.in_vocab(c)]
    
            # Find the candidate correction with the highest probability;
            # if no candidate has non-zero probability, or there are no
            # candidates, give up and output the original target word as
            # the correction.
            best_prob = float("-inf")
            best_correction = target_word
            for ivc in iv_candidates:
                ivc_log_probA = lm.log_prob(ivc, previous_word)

                ivc_log_probB = lm.log_prob(next_word, ivc)
            
                ivc_log_prob = ivc_log_probA + ivc_log_probB
            
            
                if ivc_log_prob > best_prob:
                    best_prob = ivc_log_prob
                    best_correction = ivc

            
            print(best_correction)
    else:
        # interpolation 
        Lambda = 0.25

        lm = InterpolatedLM(train_corpus, Lambda)
        for line in open(predict_corpus):
            # Split the line on a tab; get the target word to correct and
            # the sentence it's in
            target_index,sentence = line.split('\t')
            target_index = int(target_index)
            sentence = sentence.split()
            target_word = sentence[target_index]
            previous_word = sentence[target_index - 1]
            next_word = sentence[target_index + 1]
            # Get the in-vocabulary candidates (this starter code only
            # considers deletions)
            candidates = distance_one_edits(target_word)
            iv_candidates = [c for c in candidates if lm.uniLM.in_vocab(c)]
    
            # Find the candidate correction with the highest probability;
            # if no candidate has non-zero probability, or there are no
            # candidates, give up and output the original target word as
            # the correction.
            best_prob = float("-inf")
            best_correction = target_word
            for ivc in iv_candidates:
                ivc_log_probA = lm.log_prob(ivc, previous_word)

                ivc_log_probB = lm.log_prob(next_word, ivc)
               
                ivc_log_prob = math.log(ivc_log_probA * ivc_log_probB)
                
            
                if ivc_log_prob > best_prob:
                    best_prob = ivc_log_prob
                    best_correction = ivc

            
            print(best_correction)

  
    









