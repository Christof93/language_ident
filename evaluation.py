def accuracy(coll1, coll2):
    same = 0
    assert(len(coll1) == len(coll2))
    for p1, p2 in zip(coll1, coll2):
        if p1==p2:
            same+=1
    return same / len(coll1)