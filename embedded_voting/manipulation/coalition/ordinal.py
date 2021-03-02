
from embedded_voting.manipulation.coalition.general import ManipulationCoalition


class ManipulationCoalitionExtension(ManipulationCoalition):

    def __init__(self, profile=None, extension=None):
        self.profile_ = profile
        self.extension = extension

    def __call__(self, r):
        self.rule_ = r
        self.extended_rule = self.extension.set_rule(r)
        self.extended_rule(self.profile_)
        self.winner_ = self.extended_rule.winner_
        self.welfare_ = self.rule_(self.profile_).welfare_
        self.delete_cache()
        return self

    def manipulation_triviale(self, cand, verbose=False):

        voters_interested = []
        for i in range(self.profile_.n):
            score_i = self.profile_.scores[i]
            if score_i[self.winner_] < score_i[cand]:
                voters_interested.append(i)

        if verbose:
            print("%i voters interested to elect %i instead of %i" % (len(voters_interested), cand, self.winner_))

        old_profile = self.profile_.scores.copy()
        for i in voters_interested:
            self.profile_.scores[i][self.winner_] = -1
            self.profile_.scores[i][cand] = 2

        new_winner = self.extended_rule(self.profile_).winner_
        self.profile_.scores = old_profile

        if verbose:
            print("Winner is %i" % new_winner)

        return (new_winner == cand)
