def print_library_versions():
    import numpy
    import scipy
    import cloudpickle
    import pandas
    import sklearn
    import matplotlib
    import tkinter
    import threading
    print("===== Library Versions =====")
    print("Python:", __import__("sys").version.split()[0])
    print("numpy:", numpy.__version__)
    print("scipy:", scipy.__version__)
    print("cloudpickle:", cloudpickle.__version__)
    print("pandas:", pandas.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("============================\n")
    
print_library_versions()