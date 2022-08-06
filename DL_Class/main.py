import NN
import CNN
import AutoEncoder
import VariationalAutoEncoder
from LSTM import Twitter_LSTM

if __name__ == '__main__':
    # NN.run()
    # CNN.run()
    # AutoEncoder.run()
    # VariationalAutoEncoder.run()
    twitter_work = Twitter_LSTM()
    twitter_work.run()
