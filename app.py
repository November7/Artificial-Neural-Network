from NeuralNetwork import MLP
import os

        

if __name__ == "__main__":

    # inputData = [[0,0,0],
    #              [0,0,1],
    #              [0,1,0],
    #              [0,1,1],
    #              [1,0,0],
    #              [1,0,1],
    #              [1,1,0],
    #              [1,1,1],
    #              [1,2,1]]

    # outputData = [[0],[1],[0],[1],[0],[0],[1],[0],[1]]

    # inputData = [[0],[1]]
    # outputData = [[1],[0]]

    inputData = [[0,0,0],
                [1,0,0],
                [0,1,0],
                [1,0,1],
                [1,1,1],
                [1,1,0],
                [0,0,1]]

    outputData = [[0],[1],[1],[1],[0],[1],[1]]

    

    os.system('cls')
    size = len(inputData[0])

    #network = MLP([size,5,len(outputData[0])])
    network = MLP()

    network.addLayer(size,activationFunction="relu")
    network.addLayer(12,activationFunction="relu")
    network.addLayer(len(outputData[0]),activationFunction="relu")


    epoches = []
    errors = []


    epoch = 0
    progress = 0
    terr = .1

    while True:
        err = 0
        for data,out in zip(inputData,outputData):
            network.setInputs(data)
            network.fwdPropagation()
            network.bckPropagation(out)
            err += network.calcError(out)
        if err < terr: break

        progress = 100*terr/err
        if epoch % 100 == 0:
            print(f"\rErr {err:.10f}, Progress: {progress:.2f}% ",end="")
            epoches.append(epoch)
            errors.append(err)
        epoch+=1



    print()
    network.printNetwork()

    testData = []
    while True:
        testData.clear()
        for i in range(size):
            testData.append(int(input(f"Enter {i+1} value: ")))


        network.setInputs(testData)
        print(network.fwdPropagation())








