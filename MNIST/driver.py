# CHRISTOPHER HUNT
# driver.py

from network import Network

def main():
    size = (784,15,15,10)
    net = Network(size)

    net.selectionMenu()

if __name__ == "__main__":
    main()