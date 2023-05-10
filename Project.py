from GUI import Draw
import tkinter as tk

def main():
    modelFilePath = './4p80Deep/Model/'
    # leNet = LeNet(10, modelFilePath)

    root = tk.Tk()
    p = Draw(root, 'Number Canvas', modelFilePath)
    root.mainloop()
    

if __name__ == '__main__':
    main()
