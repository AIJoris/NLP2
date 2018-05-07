def viterbi2(obs, states, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": emit_p[st][obs[0]], "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for a_j, st in enumerate(states):
            #TODO fix trans_p --> len state-prevstate
            #TODO change into for loop for easeness
            #TODO use enumerate
            # max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            max_tr_prob = 0
            p=0
            for a_j_prev, prev_st in enumerate(states):
                print(st,prev_st,a_j,a_j_prev)
                print(len(states)-1)
                try:
                    p=V[t-1][prev_st]["prob"]*trans_p[len(states)-1][a_j-a_j_prev]
                except KeyError: pass
                # p=V[t-1][prev_st]["prob"] * trans_p[][]
                if p>max_tr_prob:
                    max_tr_prob=p

            for prev_st in states:
                try:

                    if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                        max_prob = max_tr_prob * emit_p[st][obs[t]]
                        V[t][st] = {"prob": max_prob, "prev": prev_st}
                        break
                except KeyError: pass

    # for line in dptable(V):
    #     print(line)

    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break

    print(V)

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
