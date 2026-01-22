# Compatibility shim for Python 3.13
# Streamlit expects imghdr but it was removed in Python 3.13

def what(file, h=None):
    return None
