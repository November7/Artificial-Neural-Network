from NeuralNetwork import MLP
import os
         

if __name__ == "__main__":

     #network.printNetwork()
    # inputData = [[0,0,0],
    #              [0,0,1],
    #              [0,1,0],
    #              [0,1,1],
    #              [1,0,0],
    #              [1,0,1],
    #              [1,1,0],
    #              [1,1,1]]

    # outputData = [[0],[1],[0],[1],[0],[0],[1],[0]]

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

    network.addLayer(size)
    network.addLayer(4)
    network.addLayer(len(outputData[0]))

    epoches = []
    errors = []
    epoch = 0
    progress = 0
    terr = .05
    while True:
        err = 0
        for data,out in zip(inputData,outputData):
            network.setInputs(data)
            network.fwdPropagation()
            network.bckPropagation(out)
            err += network.calcError(out)
        if err < terr: break

        progress = 100*terr/err
        if epoch % 500 == 0:
            print(f"\rProgress: {progress:.2f}% ",end="")
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








