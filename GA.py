
creator.create("FitnessMax" , base.Fitness , weights = (-1.0 , -1.0))
creator.create("Individual" , list , fitness = creator.FitnessMax , v_log = [])


def initial():
    return [random.randint(1 , 10) * 2.5 , random.randint(1 , 10) * 2 , random.randint(1 , 10) * 0.2 ,
            (random.randint(1 , 10)) / 10.0 , random.randint(1 , 10) * 2 , random.randint(1 , 10) * 20 ,
            random.randint(1 , 10) * 3]

    #
    # pop = np.zeros((nb_individual,genes))
    # for i in range(0,2,1):
    #     var = [random.randint(1,10)*2, random.randint(1,20)*0.5, random.randint(1,20)*5, (random.randint(1,10))/10.0, random.randint(1,10)*4]
    #     pop[i][:] += var
    # print pop


toolbox = base.Toolbox()

#                      Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool" , initial)

#                         Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')


toolbox.register("individual" , tools.initIterate , creator.Individual , toolbox.attr_bool)

# define the population to be a list of individuals
toolbox.register("population" , tools.initRepeat , list , toolbox.individual)


# the goal ('fitness') function to be maximized
def evalOneMax(individual):

    spike_rt_before = []

    spike_t_before =[]

    spike_rt_after = []

    spike_t_after = []

    IDX , T = DVS(path+'/0900.mat')
    Input = [IDX , T]
    # print 'learning IDX:', IDX, 'learning T:',T, len(IDX)
    KC2KC = np.zeros(NB_KC * (NB_KC - 1))
    mb = mb_le(In = Input , mode = 0 , pn2kc = PN2KC , kc2kc = KC2KC , pars = individual)
    # mb.run_sim()
    # brian_plot(mb.PN_STM)
    # show()

    KC2KC = mb.S_kc2kc.w

    MB = {}

    i = 0
    INPUT = [0] * 15

    for f in files:
        if not os.path.isdir(path+'/'+f):
            f_name = os.path.splitext(f)[0]

            IDX , T = DVS(path+'/'+f)
            INPUT[i] = [IDX.tolist() , T.tolist()]

            MB['mb' + str(f_name)] = mb_le(In = INPUT[i] , mode = 1 , pn2kc = PN2KC , kc2kc = KC2KC , pars = individual)

            ##record the rate of spikes of EN and EN_1

            if sum(MB['mb' + str(f_name)].EN_SPM.t / ms) == 0:
                spike_rt_before.append(0.0)
            else:
                spike_rt_before.append(np.true_divide(MB['mb' + str(f_name)].EN_SPM.num_spikes,max(MB['mb' + str(f_name)].input[1]/ms)))

            spike_t_before.append((MB['mb' + str(f_name)].EN_SPM.t /ms).tolist())

            if sum(MB['mb' + str(f_name)].EN_1_SPM.t / ms) == 0:
                spike_rt_after.append(0.0)
            else:
                spike_rt_after.append(np.true_divide(MB['mb' + str(f_name)].EN_1_SPM.num_spikes, max(MB['mb' + str(f_name)].input[1]/ms)))

            spike_t_after.append((MB['mb' + str(f_name)].EN_1_SPM.t / ms).tolist())


    individual.v_log = [spike_rt_before,spike_rt_after, spike_t_before, spike_t_after]

    f1 = sum(map(lambda x:  np.square(x-0.4), spike_rt_before[0:8]))
    f2 = 0
    for z in range(8):
        f2 = f2 + np.square(spike_rt_after[z] - 0.4 - (0.05*z))
    # f3 = 0
    # for y in range(9):
    #     f3 = f3 + np.square( max(mbon_before[y+3][0]) - 30)
    #     print max(mbon_before[y+3][0]) , len(mbon_before[y+3][0])

    # subplot(211)
    # brian_plot(MB['mb0900'].EN_STM)
    # subplot(212)
    # brian_plot(MB['mb0900'].EN_1_STM)
    # show()

    return f2, f1


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate" , evalOneMax)

# register the crossover operator
toolbox.register("mate" , tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate" , tools.mutGaussian , indpb = 0.1)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select" , tools.selTournament , tournsize = 3)


# ----------

def main(args):


    random.seed(64)
    # nb_individual = 2
    # genes = 5
    # pop = np.zeros((nb_individual,genes))
    # for i in range(0,2,1):
    #     var = [random.randint(1,10)*2, random.randint(1,20)*0.5, random.randint(1,20)*5, (random.randint(1,10))/10.0, random.randint(1,10)*4]
    #     pop[i][:] += var

    # pop.tolist()
    # create an initial population of 50 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n = 20)
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB , MUTPB = 0.5 , 0.5

    print("Start of evolution") , 'at:' , time.asctime(time.localtime(time.time()))

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate , pop))
    for ind , fit in zip(pop , fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    EA_log = []
    # Begin the evolution
    while g < 40:
        # A new generation
        population = {}
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop , len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone , offspring))

        # Apply crossover and mutation on the offspring
        for child1 , child2 in zip(offspring[::2] , offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1 , child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant , 2 , 1)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate , invalid_ind)
        for ind , fit in zip(invalid_ind , fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        Best_ind = tools.selBest(offspring , 1)[0]
        print("Best individual is %s, %s" % (Best_ind , Best_ind.fitness.values))

        EA_log.append([g , Best_ind , [Best_ind.fitness.values] , Best_ind.v_log])
        #
        #
        # for i in range(0,len(pop),1):
        #     population['individual_'+str(i)] = [pop[i]]
        #     population['individual_'+str(i)+'_fitness'] = pop[i].fitness.values
        #     population['individual_'+str(i)+'_v_log'] = pop[i].v_log
        # # GA_log['generation'+str(g)]  = pop
        # # GA_log['generation'+str(g)+'fitness'] = fits
        # # GA_log['generation'+str(g)+'v'] = vlog
        # GA_log.append(population)

        # f = open(str(args.output)+'/EA_log.log', 'w')
        # json.dump(EA_log , f)
        # print("-- Generation %i --" % g) , 'is loaded'

        # # print 'GA:',  type(GA_log),GA_log
        #
        # file = open('GA.pickle','w+')
        # pickle.dump(GA_log, file, 1)

        # json_str = json.dumps(GA_log , indent = 1)
        # with open('GA.json' , 'w') as json_file:
        #     json_file.write(json_str)
        # f = open(str(args.output) + '/EA_log' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.json' ,
        #          'w+')
        # json.dump(EA_log , f)

    # with open("/home/le/Disk/ea_log/GA_3.json" , "w") as f:
    #     json.dump(EA_log , f)
    # print("-- Generation %i --" % g) , 'is loaded'
    f = open(str(args.output)+'/EA_log'+ str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+'.json', 'w')
    json.dump(EA_log , f)
    print("-- Generation %i --" % g) , 'is loaded'

    print("-- End of (successful) evolution --")

    print time.asctime(time.localtime(time.time()))
    best_ind = tools.selBest(pop , 1)[0]
    print("Best individual is %s, %s" % (best_ind , best_ind.fitness.values))


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    path = args.input

    files = os.listdir(path)
    files.sort()
    main(args)