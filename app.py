from detect_button import detect_button
import sys 
import os 

def main():
    if (len(sys.argv) < 2): return
    path = os.path.join(os.curdir, sys.argv[1])
    if (os.path.isfile(path)):
        detect_button(path)
    
    elif (os.path.isdir(path)):
        for entry in os.scandir(path):
            if (entry.path.endswith('jpg') or entry.path.endswith('png')):
                detect_button(entry.path)



if __name__ == "__main__":
    main()  