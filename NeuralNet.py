import random

e=(1+1/10000000)**10000000

def sigmoid(x):
    try:
        return 1/(1+e**(-1*x))
    except OverflowError:
        return 0

def Dsig(x):
    return (sigmoid(x))*(1-sigmoid(x))

def DsigV(v):
    return [Dsig(x) for x in v]

def cost(v1, v2):
    cost=0
    for i in range(len(v1)):
        cost+=(v1[i]-v2[i])**2
    cost/=len(v1)
    return cost

def dot(v1, v2):
    dot=0
    for i in range(len(v1)):
        dot+=v1[i]*v2[i]
    return dot

def multiply(v1, v2):
    result=[]
    for i in range(len(v2)):
        toAdd=[]
        for j in range(len(v1)):
            toAdd.append(v2[i]*v1[j])
        result.append(toAdd)
    return result

def multiplyC(v1, v2):
    return [v1[i]*v2[i] for i in range(len(v1))]

def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def multiplyMV(m, v):
    result=[]
    for i in m:
        result.append(dot(i, v))
    return result

def create_mini_batch(data):
    half=round(len(data)/2)
    batch=[]
    data=data[:]
    for i in range(half):
        toAdd=random.choice(data)
        batch.append(toAdd)
        data.remove(toAdd)
    return batch


class Layer:
    def __init__(self, inputNodes, outputNodes):
        self.weights=[]
        self.biases=[]
        self.inputNodes=inputNodes
        self.outputNodes=outputNodes
        self.weightedInputs=[]
        self.outputs=[]
        self.inputs=[]
        
        #default params
        for i in range(inputNodes):
            listToAdd=[]
            for i in range(outputNodes):
                listToAdd.append(random.random()*2-1)
            self.weights.append(listToAdd)
        for i in range(outputNodes):
            self.biases.append(random.random()*4-2)

    def output(self, inputs):
        outputs=self.biases[:]
        for i in range(self.inputNodes):
            for j in range(self.outputNodes):
                weight=self.weights[i][j]
                outputs[j]+=inputs[i]*weight
        self.weightedInputs=outputs[:]
        for i in range(self.outputNodes):
            outputs[i]=sigmoid(outputs[i])
        self.outputs=outputs
        self.inputs=inputs
        return outputs

class Network:
    def __init__(self, layer_sizes):
        self.layers=[]
        self.learnRate=0.3
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def output(self, inputs):
        outputs=inputs[:]
        for i in self.layers:
            outputs=i.output(outputs)
        return outputs

    def avg_cost(self, data):
        result=0
        for point in data:
            result+=cost(point[1], self.output(point[0]))
        return result/len(data)

    def fast_gradient_descent(self, dataPoint):
        output=self.output(dataPoint[0])
        error=[output[i]-dataPoint[1][i] for i in range(len(output))]
        weight_changes=[[] for i in self.layers]
        bias_changes=weight_changes[:]
        for i in range(1, len(self.layers)+1):
            index=len(self.layers)-i
            layer=self.layers[index]
            weight_changes[index]=multiply(error, layer.inputs)
            bias_changes[index]=error[:]
            updated_weights=[[] for i in layer.weights]
            for i in range(len(weight_changes[index])):
                for j in range(len(weight_changes[index][i])):
                    updated_weights[i].append(layer.weights[i][j]-weight_changes[index][i][j]*self.learnRate)

            error=multiplyC(multiplyMV(updated_weights,error), DsigV(layer.inputs))
        return [weight_changes, bias_changes]

    def complete_gradient_descent(self, data):
        gradient_results=[]
        for i in data:
            gradient_results.append(self.fast_gradient_descent(i))
            
        gradient_average_W=gradient_results[0][0][:]
        for i in range(1, len(gradient_results)):
            wChanges=gradient_results[i][0]
            for index in range(len(wChanges)):
                for a in range(len(wChanges[index])):
                    for j in range(len(wChanges[index][a])):
                        gradient_average_W[index][a][j]+=wChanges[index][a][j]
                    
        gradient_avg_B=gradient_results[0][1][:]
        for i in range(1, len(gradient_results)):
            BChanges=gradient_results[i][1]
            for index in range(len(BChanges)):
                for j in range(len(BChanges[index])):
                    gradient_avg_B[index][j]+=BChanges[index][j]
                    
        for i in range(len(self.layers)):
            layer=self.layers[i]
            B_Changes=gradient_avg_B[i]
            w_Changes=gradient_average_W[i]
            for j in range(len(layer.weights)):
                for k in range(len(layer.weights[j])):
                    layer.weights[j][k]-=w_Changes[j][k]*self.learnRate*(1/len(data))
            for j in range(len(layer.biases)):
                layer.biases[j]-=B_Changes[j]*self.learnRate*(1/len(data))
                    

    def apply_gradients(self, gradients):
        count=0
        for ly in range(len(self.layers)):
            layer=self.layers[ly]
            for weight_list in range(len(layer.weights)):
                for weight in range(len(layer.weights[weight_list])):
                    layer.weights[weight_list][weight]-=gradients[count]*self.learnRate
                    count+=1
            for bias in range(len(layer.biases)):
                layer.biases[bias]-=gradients[count]*self.learnRate
                count+=1
            
    def slow_gradient_descent(self, data):
        default_cost=self.avg_cost(data)
        gradients=[]
        for layer in self.layers:
            for weight_list in range(len(layer.weights)):
                for weight in range(len(layer.weights[weight_list])):
                    layer.weights[weight_list][weight]+=10**(-4)
                    gradient=(self.avg_cost(data)-default_cost)*10000
                    layer.weights[weight_list][weight]-=10**(-4)
                    gradients.append(gradient)
            for bias in range(len(layer.biases)):
                layer.biases[bias]+=10**(-4)
                gradient=(self.avg_cost(data)-default_cost)*10000
                layer.biases[bias]-=10**(-4)
                gradients.append(gradient)
        self.apply_gradients(gradients)

    def train(self, iterations, data):
        for i in range(iterations):
            self.complete_gradient_descent(create_mini_batch(data))

if __name__=="__main__":
    data=[[[0, 0], [1, 0]],
      [[1, 0], [0, 1]]]

    #works for 1 hidden layer, sometimes does not work for more than 2 hidden layers
    network=Network([2, 5, 6, 2])
    network.train(3000, data)
    print(cost(network.output([0, 0]), [1, 0]))
    print(cost(network.output([1, 0]), [0, 1]))
    #new data
    print(cost(network.output([0, 1]), [1, 0]))
    print(cost(network.output([1, 1]), [0, 1]))
        
