import os
import socket


def check_port(default_port=29500):
    curr_port = os.environ['MASTER_PORT'] if 'MASTER_PORT' in os.environ else default_port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('', curr_port))
        s.close()
    except socket.error:
        s.bind(('', 0))
        addr, curr_port = s.getsockname()
        s.close()
    return curr_port


if __name__ == '__main__':
    print(check_port())
