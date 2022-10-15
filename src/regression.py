import pandas as pd
import numpy as np

class linear_regression() :
    def __init__(self,data : pd.DataFrame , label : pd.DataFrame, epsilon) -> None:
        self.data = data 
        self.label = label.values.reshape((label.shape[0],1))
        self.data["bias"] = 1
        self.data = self.data.values
        self.features_num = self.data.shape[1]
        self.w = np.zeros((self.features_num, 1))
        self.epsilon = epsilon
        
    
    def fit(self, eta) :
        p_cost = 0
        while True :
            cost = self.train(eta=eta)
            print(f"p_cost is : {p_cost} and cost is : {cost[0,0]}")
            if abs(p_cost - cost[0,0]) < self.epsilon :
                break
            p_cost = cost[0,0]
        print(cost[0,0])

        

    def train(self, eta) :
        errors =  self.error()
        return self.w_optimization(errors,eta)

    def w_optimization(self, errors : np.ndarray, eta) :
        self.w += eta*2*self.data.T.dot(errors)/float(self.data.shape[0])
        return self.cost(self.data, self.w)

    def error(self) :
        return self.label - self.model(self.data, self.w)

    def model(self, X, nw) :
        return np.dot(X , nw)
    
    def cost(self, X, nw) : 
        cost = self.error().T.dot(self.error())/float(self.features_num) 
        return cost

    # to predict with model we need to add a bias section to the x parameter in test data down below : 
    def predict(self, x) :
        return np.dot(x.T, self.w)