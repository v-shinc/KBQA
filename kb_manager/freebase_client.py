# from __future__ import absolute_import


from gen_py.freebase_service import FreebaseService
from gen_py.freebase_service.ttypes import *
from gen_py.freebase_service.constants import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

class FreebaseClient:
    def __init__(self):
        try:
          # Make socket
          self.transport = TSocket.TSocket('127.0.0.1', 8888)

          # Buffering is critical. Raw sockets are very slow
          self.transport = TTransport.TBufferedTransport(self.transport)

          # Wrap in a protocol
          protocol = TBinaryProtocol.TBinaryProtocol(self.transport)

          # Create a client to use the protocol encoder
          self.client = FreebaseService.Client(protocol)

          # Connect!
          self.transport.open()

        except Thrift.TException, tx:
          print "%s" % (tx.message)

    def get_subgraph(self, subject):
        print "receive requestion", subject
        ret = self.client.get_subgraph(subject)
        print ret
        return ret

    def close(self):
        self.transport.close()

if __name__ == '__main__':
    client = FreebaseClient()
    print client.get_subgraph('m.0mbq3')
    client.close()
