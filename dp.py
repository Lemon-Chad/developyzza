import shell
import devp
import sys
import os
from standlibs import pretzel, time

pretzel.init("tk_pretzel")
time.init("time")

if os.path.exists('dp_old.exe'):
    os.remove('dp_old.exe')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        if args[0] == 'help':
            print("dp -> Open venv\ndp help -> List commands\ndp docs -> Link to documentation\ndp <file> -> Run file")
        elif args[0].endswith('.devp'):
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            fn = args[0]
            if os.path.exists(fn):
                result, error = devp.run(fn, open(fn, 'r').read())
                if error:
                    print(error.as_string())
            else:
                print("File does not exist.")
        elif args[0] == 'docs':
            input("Documentation: https://bit.ly/3vM8G0a")
    else:
        shell.shell()
