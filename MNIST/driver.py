# CHRISTOPHER HUNT
# driver.py

from network import Network

def main():
    size = (784,15,10)
    net = Network(size)

    print(net.weights[0].shape)

    #net.selectionMenu()

if __name__ == "__main__":
    main()