from embedded_voting.embeddings.generator import RandomEmbeddings
from embedded_voting.profile.impartial import ImpartialCulture
from embedded_voting.profile.moving import MovingEmbeddings
from embedded_voting.scoring.singlewinner.svd import SVDNash, SVDSum


def test_impartial():
    ImpartialCulture(10, 5)


def test_moving():
    embeddings = RandomEmbeddings(100, 3)()
    moving_embs = MovingEmbeddings(embeddings)
    ratings = ImpartialCulture(100, 4)
    moving_embs(SVDNash(), ratings)
    moving_embs(SVDSum())


