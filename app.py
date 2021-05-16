from detect_button import detect_button
import sys 
import os 

def main():
    if (len(sys.argv) < 2): return
    detect_button(os.path.join(os.curdir, 'images', sys.argv[1]))



if __name__ == "__main__":
    main()  