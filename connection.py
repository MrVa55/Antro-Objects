# connection.py
import line_packet

class Connection:
    '''This class wraps a socket connection object.'''
    PACKET_SIZE = 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""
        self.conn.setblocking(True)

    def send(self, line):
        '''Send a line through the socket, ensuring not to send the same line twice.'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        '''Receive lines from the socket.'''
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        '''Receive audio from the socket in a non-blocking manner.'''
        r = self.conn.recv(self.PACKET_SIZE)
        return r
