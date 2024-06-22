# Neural-Network-from-scratch
---

A[0] : First/input layer = x

z1 : Unactivated first layer = W * A[0] + b[1] /// Weights*Input + bias  or z1 = (w*x) + b

Activation Function : Sigmoid function // ReLu {rectified linear unit} : {

    x if x>0
    
    0 if x<=0
    
}

A[1] : Activated First Layer = ReLu(z1)

z2 : Unactivated second Layer = W2 * A[1] + b[2]

Activation Function = Softmax : {

    e[z_i]/Summation{j=1 ~ K} e[z_j]
    
}

A[2] = softmax(z2) {Probabilities}

BackProp to fix biases and weights--

dz2 : Error of second layer = A[2] - Y

dW2 : Derivative of cost function w.r.t. weights in layer 2 

db[2] : Average of absolute error for layer 2

dz1 : 

dw1 :

db[1] :

Update params

W1 = W1 - a*dW1

a : Learning Rate {Hyperparameter : Set by Coder(Us)}

// After all this run whole shit again


