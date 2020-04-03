import embed
from embed.embed import embed_data
from char.word2idx import word2idx

def _gen_bc_data():
    embed_data(save=True)
    word2idx(maxwlen=10, save=True)
if __name__ == '__main__':
    _gen_bc_data()
