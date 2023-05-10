from LeNet import LeNet
from GUI import Draw
import tkinter as tk

def main():
    ch = None
    modelFilePath = './4p80Deep/Model/'

    while (ch != '3'):
        ch = input('1: Train new model\n2: Test current Model\n3: Exit\n> ')

        if ch == '1':
            try:
                numEpochs = int(input('Number of epochs to train for: '))
                
                leNet = LeNet(numEpochs, modelFilePath)

            except:
                print('Integers only.')

        elif ch == '2':
            root = tk.Tk()
            p = Draw(root, 'Number Canvas', modelFilePath)
            root.mainloop()
    

if __name__ == '__main__':
    main()
