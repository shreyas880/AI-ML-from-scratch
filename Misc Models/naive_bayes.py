import numpy as np

class NaiveBayes:
    # def fit(self, X, y):
    #     num_samples, num_features = X.shape
    #     self._classes = np.unique(y)
    #     num_classes = len(self._classes)

        # # mean, variance and prior for each class
        # self._mean = np.zeros(num_classes, num_features)
        # self._var = np.zeros(num_classes, num_features)
        # self._priors = np.zeros(num_classes)

        # for idx, c in enumerate(self._classes):
        #     X_c = X[y==c]
        #     self._mean[idx, :] = X_c.mean(axis=0)
        #     self._var[idx, :] = X_c.var(axis=0)

        #     self._priors[idx] = X_c.shape[0]/float(num_samples)

    def fit(self, X, y):
        # finding P(y)
        # Class is every unique value that y can take
        num_samples, num_features = X.shape

        self.classes = np.unique(y)

        num_classes = len(self.classes)

        # mean, variance and prior for each class
        self.mean = np.zeros((num_classes, num_features))
        self.var = np.zeros((num_classes, num_features))
        self.priors = np.zeros(num_classes)

        # Returns only the classes

        # mean of x-values for each unique y
        # if self.classes = ['Label 1', 'Label 2', 'label 3']
        # idx, cl -> (0, 'Label 1') and so on

        for idx, cl in enumerate(self.classes):
            # X_c - X values of a class C
            X_c = X[y==cl]

            # for i in range(num_features):
            #     self.mean[idx, i] = np.mean(X_c[:, i])
            # The above code is essentially what is being done, but just put in a more efficient way
            self.mean[idx, :] = np.mean(X_c, axis = 0)
            self.var[idx, :] = np.var(X_c, axis = 0)

            # X_c.shape[0] returns the occurance of that particular class
            self.priors[idx] = X_c.shape[0]/num_samples
            

    def predict(self, X):
        y_pred = []

        for x in X:
            posteriors = []

            for idx, cl in enumerate(self.classes):
                # Calculate prior with log(P(y))
                # P(y) = # occurance of y/ no. of unique y
                prior = np.log(self.priors[idx])

                prob_x = np.sum(np.log(self._PDF(idx, x)))
                
                posterior = prob_x + prior

                posteriors.append(posterior)

            y_pred.append(self.classes[np.argmax(posteriors)])

        return np.array(y_pred)

    def _PDF(self, idx, x):
        mean = self.mean[idx]
        var = self.var[idx] + 1e-9
        
        num = np.exp(-((x - mean)**2)/(2*var))
        den = np.sqrt(2*np.pi*var)

        return num/den

    

    # P(y|X) = sigma: log(P(x1|y)) + log(P(y))
    # P(y|X): posterior
    # y is the argmax of all posteriors
    # P(y): prior probability --> Frequency of each class
    # P(xi|y): conditional probability for a class -> Modelled w/ gaussian probability distribution


