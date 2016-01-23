import sys
from .Launcher import Launcher

def main():
    l = Launcher()
    l(sys.argv[1:])
    return None

if __name__ == '__main__':
    main()

