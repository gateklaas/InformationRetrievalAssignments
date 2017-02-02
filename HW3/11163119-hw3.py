# Name: Klaas Schuijtemaker
# Date: February 2 2017
# Student nr.: 11163119

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query as q
import math
from multiprocessing import Process

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 200
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

# Get DCG@k of a list with relevances
def get_dcg(rel, k):
    return sum((2**rel[i] - 1) / math.log(i + 2, 2) for i in range(k))

# Get NDCG@k of a list with relevances
def get_ndcg(rel, k):
    dcg = get_dcg(rel, k)
    max_dcg = get_dcg(sorted(rel, reverse=True), k)
    return 0 if max_dcg == 0 else dcg / max_dcg

# Step 3: Cost = 1/2(1-Sij)(f(u)-f(v)) + log(1+exp(f(v)-f(u)))
def lambda_loss(f, y):
    Y = T.tile(y, (y.shape[0],1))
    S = T.sgn(Y.T - Y)
    F = T.tile(f, (f.shape[1],1))
    F = F.T - F
    C = 0.5 * (1 - S) * F + T.log(1 + T.exp(F.T))
    return C

class LambdaRankHW:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, algorithm='pointwise'):
        self.feature_count = feature_count
        self.algorithm = algorithm
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)

    def train_with_queries(self, train_queries, num_epochs):
        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 10 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass
    
    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    # Get NDCG@k for a given query
    def ndcg(self, query, k=10):
        score = self.score(query)
        labels = query.get_labels()
        ranking = [x[1] for x in sorted(zip(score, labels), key=lambda x: x[0], reverse=True)]
        return get_ndcg(ranking, k)


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim",input_dim, "output_dim",output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=NUM_HIDDEN_UNITS,
            nonlinearity=lasagne.nonlinearities.tanh,
        )


        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")

        # 
        if self.algorithm == 'pointwise':
            loss_train = lasagne.objectives.squared_error(output,y_batch)
        
        # Step 3 & 6
        elif self.algorithm == 'pairwise' or self.algorithm == 'listwise':
            loss_train = lambda_loss(output,y_batch)
        
        else:
            raise Error()
        
        loss_train = loss_train.mean()

        # regularization would make it even slower

        all_params = lasagne.layers.get_all_params(output_layer)
        updates = lasagne.updates.adam(loss_train, all_params)

        score_func = theano.function(
            [X_batch],output_row_det,
            allow_input_downcast=True,
        )

        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            allow_input_downcast=True,
        )

        print "finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )
        
    # Step 3: Lambda = 1/2(1-Suv) - 1/(1+exp(f(u)-f(v)))
    def lambda_function(self, y, f):
        I = range(len(y))
        L = [[0.5 * (1 - np.sign(y[i] - y[j])) - 1 / (1 + np.exp(f[i] - f[j])) for i in I] for j in I]
        l = [sum(L[i][j] for j in I) - sum(L[j][i] for j in I) for i in I] # aggregate
        return np.array(l)
    
    # Step 6: Lambda = -1 / (1+exp(f(u)-f(v))) * NDCG@max
    def lambda_ndcg_function(self, y, f):
        ranking = [x[1] for x in sorted(zip(f, y), key=lambda x: x[0], reverse=True)]
        ndcg = get_ndcg(ranking, len(ranking))
        I = range(len(y))
        L = [[-1 / (1 + np.exp(f[i] - f[j])) * ndcg for i in I] for j in I]
        l = [sum(L[i][j] for j in I) - sum(L[j][i] for j in I) for i in I] # aggregate
        return np.array(l)
    
    # Step 3: call lambda_function()
    def compute_lambdas_theano(self, query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result
    
    # Step 6: call lambda_ndcg_function()
    def compute_lambdas_ndcg_theano(self, query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_ndcg_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):

        # 
        if self.algorithm == 'pointwise':
            X_train.resize((min(BATCH_SIZE,len(labels)), self.feature_count),refcheck=False)
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        
        # Step 3: Call compute_lambdas_theano()
        elif self.algorithm == 'pairwise':
            lambdas = self.compute_lambdas_theano(query,labels)
            lambdas.resize((BATCH_SIZE, ))
            X_train.resize((BATCH_SIZE, self.feature_count),refcheck=False)
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        
        # Step 6: Call compute_lambdas_ndcg_theano()
        elif self.algorithm == 'listwise':
            lambdas = self.compute_lambdas_ndcg_theano(query,labels)
            lambdas.resize((BATCH_SIZE, ))
            X_train.resize((BATCH_SIZE, self.feature_count),refcheck=False)
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
            
        else:
            raise Error()
        
        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()

                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                batch_train_losses.append(batch_train_loss)
                

            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

# Step 5: Cross validation on hyper-parameter: num_hidden_units
def crossvalidation_num_hidden_units(file_name, feature_count, num_hidden_units=[100,300,500], algorithm='pointwise', num_epochs=1, ndcg_k=10):

    with open(file_name, 'w') as fp:
        fp.write('Cross-validation on num-hidden-units. feature_count: %d, algorithm: %s, num_epochs:%d\n' % (feature_count, algorithm, num_epochs))

        for nhu in num_hidden_units:
        
            ndcg_mean2 = 0
            for fold in [1,2,3,4,5]:
                
                NUM_HIDDEN_UNITS = nhu
                ranker = LambdaRankHW(feature_count, algorithm=algorithm)
                
                # Train on training-set
                queries = q.load_queries("HP2003/Fold" + str(fold) + "/train.txt", feature_count)
                ranker.train_with_queries(queries, num_epochs)
            
                # Validate on validation-set
                queries = q.load_queries("HP2003/Fold" + str(fold) + "/vali.txt", feature_count)

                # Calc mean NDCG@10
                ndcg_mean = 0
                for i, query in enumerate(queries):
                    ndcg = ranker.ndcg(query, k=ndcg_k)
                    ndcg_mean += ndcg
                    print 'Finished query:', i
                
                ndcg_mean = ndcg_mean / len(queries)
                ndcg_mean2 += ndcg_mean
                print 'Finished fold:', fold
            
            ndcg_mean2 = ndcg_mean2 / 5
            fp.write('Number of hidden units: %d, NDCG@%d: %f\n' % (nhu, ndcg_k, ndcg_mean2))
            print 'Finished num_hidden_units:', nhu
        print 'Finished cross-validation'

# Step 7: Evaluate ranking algorithm by calculating the mean NDCG@10
def evaluate_ranker(ranker, file_name, num_epochs=5, ndcg_k=10):

    with open(file_name, 'w') as fp:

        for fold in [1,2,3,4,5]:
        
            # Train on all traning-sets
            queries = q.load_queries("HP2003/Fold" + str(fold) + "/train.txt", ranker.feature_count)
            ranker.train_with_queries(queries, num_epochs)
            
            # Train on all validation-sets (This is allowed since the validation phase is already finished)
            queries = q.load_queries("HP2003/Fold" + str(fold) + "/vali.txt", ranker.feature_count)
            ranker.train_with_queries(queries, num_epochs)
            
            print 'finished training on fold', fold

        ndcg_mean2 = 0
        for fold in [1,2,3,4,5]:
        
            # Evaluate ranker on all test-set
            queries = q.load_queries("HP2003/Fold" + str(fold) + "/test.txt", ranker.feature_count)

            # Calc mean NDCG@10
            ndcg_mean = 0
            for i, query in enumerate(queries):
                ndcg = ranker.ndcg(query, k=ndcg_k)
                ndcg_mean += ndcg
                fp.write('  Finished test on query %d result: NDCG@%d: %f\n' % (i+1, ndcg_k, ndcg))

            ndcg_mean = ndcg_mean / len(queries)
            ndcg_mean2 += ndcg_mean
            fp.write(' Finished test on fold %d result: NDCG@%d: %f\n' % (fold, ndcg_k, ndcg_mean))

        ndcg_mean2 = ndcg_mean2 / 5
        fp.write('Final result: NDCG@%d: %f\n' % (ndcg_k, ndcg_mean2))

# Set paramameters
feature_count = 64

# First we do cross validation to find the optimal value for the hyper-parameter: num_hidden_units
crossvalidation_num_hidden_units('Cross-validation pointwise.txt', feature_count, num_hidden_units=[100,300,500], algorithm='pointwise', num_epochs=1)
crossvalidation_num_hidden_units('Cross-validation pairwise.txt', feature_count, num_hidden_units=[100,300,500], algorithm='pairwise', num_epochs=1)
crossvalidation_num_hidden_units('Cross-validation listwise.txt', feature_count, num_hidden_units=[100,300,500], algorithm='listwise', num_epochs=1)

# Now we have the optimal value for the num_hidden_units hyper-parameter.
# We set the num_hidden_units to its optimal value and then evaluate the ranking algorithm.
# The evaluation is done by calculating the mean NDCG@10 over 5-folds
NUM_HIDDEN_UNITS = 500
ranker = LambdaRankHW(feature_count, algorithm='pointwise')
evaluate_ranker(ranker, 'Evaluation pointwise.txt', num_epochs=5)

NUM_HIDDEN_UNITS = 500
ranker = LambdaRankHW(feature_count, algorithm='pairwise')
evaluate_ranker(ranker, 'Evaluation pairwise.txt', num_epochs=5)

NUM_HIDDEN_UNITS = 300
ranker = LambdaRankHW(feature_count, algorithm='listwise')
evaluate_ranker(ranker, 'Evaluation  listwise.txt', num_epochs=5)

'''

# RESULTS

The optimal values for the hyper-parameter called num_hidden_units are:
- RankNet pointwise: 500
- RankNet pairwise: 500
- LambdaRank listwise: 300

After evaluating, the mean NDCG@10 of the algorithms are
- RankNet pointwise: 0.577375
- RankNet pairwise: 0.414644
- LambdaRank listwise: 0.296635


# ANALYSIS
The evaluation results are rather dispointing. I expected the LambdaRank to have the highest evaluation score and the RankNet pairwise, to have the second highest evaluation score. Instead, the results are reversed. This suggests that a mistake was made in the implementation of the lambda function or the loss function. 

'''