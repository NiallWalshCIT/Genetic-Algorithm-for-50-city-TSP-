# Niall Walsh
# GA template for solving TSP problem
# import relevant libraries
import random
import numpy as np
import matplotlib.pyplot as plt

def initializePopulation(populationSize, numCities):
    # create an empty array 
    population = np.zeros((populationSize, numCities), dtype = int)
    # added a random shuffle of cities to the array    
    for i in range(0,populationSize):
        arr = np.arange(numCities)
        np.random.shuffle(arr)
        population[i] = arr       
    return population

def calculatePopulationFitness(population, distancesMatrix):
    # create list to store the fitness value of each individual  
    fitness = []
    for i in range(0,len(population)):
        #create a list to store values
        distance_list = []
        # assigns one row from the ppulation to be an individual
        individual = population[i]
        # create a loop to run til the second last element in indivdual
        for i in range(0,len(individual)-1):
            # find the distance between each city and append the distances to a list
            distance_list.append(distancesMatrix[int(individual[i]),int(individual[i+1])])
        # calculate the distnace between last and first cities in an individual and append it to distancelist
        distance_list.append(distancesMatrix[int(individual[len(individual)-1]),int(individual[0])])
        # added the distances and append the total to the fitness list
        fitness.append(sum(distance_list))
    #convert the values to a numpy array and assign them to fitness Values
    fitnessValues = np.asarray(fitness)
    #return fitnessValues
    return fitnessValues
  
def randomSelection(population, fitnessValues, populationSize):
    # 2% of the population
    test_size = int((populationSize / 100) * 2)
    # generate random numbers between 0 and population size based on the test size
    random_numbers = random.sample(range(0, populationSize), test_size)
    # create a list
    list_values = []
    # total distance to travel for each of the 5 random samples 
    for i in random_numbers:
        list_values.append(fitnessValues[i])
    # smallest first
    list_values.sort()
    # convert to numpy array
    list_values = np.asarray(list_values)
    # get the best and second best sequence of cities and assign it to parent 1 and 2
    parent1 = population[list(fitnessValues).index(list_values[0])]
    parent2 = population[list(fitnessValues).index(list_values[1])]
    return parent1, parent2
        
def performSinglePointCrossOver(parent1, parent2, numCities):
    # set crossover type for printing on graph
    crossovertype = 'SinglepointCrossover'
    # set child to me an empty list
    child = []
    # crossoverpoint is set as the number of cities divided by 3
    # all elements up to this point are added from parent 1
    for i in range(0,numCities/3):
        child.append(parent1[i])
    # any element not in the child is added from parent 2  
    for i in parent2:  
        if i not in child:
            child.append(i)
    return child, crossovertype
    
def performRandomCrossOver(parent1, parent2, numCities):
    # set crossover type for printing on graph
    crossovertype = 'RandomCrossover'
    # set child to me an empty list
    child =[]
    # crossover point is a random number between 0 and the number of cities.
    # Elments up to this crossover point are appened to the child from parent 1
    for i in range(0,random.randint(0,numCities)):
        child.append(parent1[i])
    # Parent 2 is iterated through and any element not in the child is appended 
    for i in parent2:  
        if i not in child:
            child.append(i)
    return child, crossovertype

def performCycleCrossOver(parent1, parent2):
    # set crossover type for printing on graph
    crossovertype = 'CycleCrossover'
    # set child to me an empty list
    child = []
    # Convert parents to list
    parent1 = list(parent1)
    parent2 = list(parent2)
    # make a list of '@' the length of parent and call it child
    child = ['@'] * len(parent1)
    # iterate through the child each time checking to see if an element of parent two is
    # in the child. Intiatlly we start with the first element of parent 2 and we find the 
    # the corresdonding index for that value in parent one and set that index in child to be
    # that element index in parent one. The second loop then uses that index of parent 2 and 
    # again iterates through the child. This continues until an element from parent 2 is in child.
    i = 0
    while parent2[i] not in child:
        city = parent2[i]
        i = parent1.index(city)
        child[i] = parent1[i]
    # we then iterate through the child and swap any '@' for the corresponding element in parent 2
    for i in range(0,len(parent1)):  
        if child[i] == '@':
            child[i] = parent2[i]        
    return child, crossovertype

def simplemutate(child):
    #Simple mutation
    child[3], child[9] = child[9], child[3]
    return child

def randomswapmutate(child, numCities):
    # create a list of two random numbers between 0 and number of cities
    rand = random.sample(range(0, numCities), 2)
    #Swap the elements of the list with the random number indexes
    child[rand[0]], child[rand[1]]= child[rand[1]], child[rand[0]]
    return child

def randomshufflemutate(child, numCities):
    # generate two random numbers between 0 and number of cities
    rand = random.sample(range(0, numCities), 2)
    child1 = []
    # append the elements of the child until the start of the shuffle
    for i in range(0,np.min(rand)):
        child1.append(child[i])
        
    # shuffle the elements of the child between the two random numbers
    childshuffle = child[np.min(rand):np.max(rand)]
    np.random.shuffle(childshuffle)
    # added the shuffled elements
    for i in range(0,len(childshuffle)):
        child1.append(childshuffle[i])
    # added elements after the shuffle   
    for i in range(np.max(rand),len(child)):
        child1.append(child[i])
    
    return child1

def main():
    # Read distances matrix from a file
    distancesMatrix = np.genfromtxt("50City.txt")
    # Calculate the number of cities
    numCities = distancesMatrix.shape[0]
    # set the number of GA trial runs
    numGAtrials = 10
    #create two lists for minimum and mean distances for each trial run
    Min_fitness_values =[]
    Mean_fitness_values = []
    # create an array to store the best sequence of each GA trial
    fit_pop = np.zeros((numGAtrials, numCities), dtype = int)
    # set the probabiltiy of mutation 
    probmutation = 0.5
    
    for trial in range(0,numGAtrials):
        # set pop size
        populationSize = 1000
        # set number of generations
        numberOfGenerations = 100
        # Generate an initial population of individuals
        population = initializePopulation(populationSize, numCities)
        # Iterate for a fixed number of generations
        for num in range(numberOfGenerations):
            # calculate fitness for all individuals in the current population
            # return a 1D arrays containing fitness of each individual
            fitnessValues = calculatePopulationFitness(population, distancesMatrix)
            # create a new population for the current generation
            newPopulation = np.zeros((populationSize, numCities), dtype = int)
            for newIndex in range(populationSize):
                # tournament selection to select parents
                parent1, parent2 = randomSelection(population, fitnessValues, populationSize)
                
                # Cross over two parents to form a child, also return crossovertype for printing on graph
                child, crossovertype = performSinglePointCrossOver(parent1, parent2, numCities)           
                
                child, crossovertype = performCycleCrossOver(parent1, parent2)
                
                child, crossovertype = performRandomCrossOver(parent1, parent2, numCities)
                
                #  Mutate if condition
                if random.uniform(0, 1) < probmutation:
                    child = simplemutate(child)
                    
                    child = randomswapmutate(child, numCities)
                    
                    child = randomshufflemutate(child, numCities)
                    
                # Add the child to the new population
                newPopulation[newIndex] = child
            # Set the new population equal to the current population
            population = newPopulation
        # calculate fitness Values for the new population 
        fitnessValues = calculatePopulationFitness(population, distancesMatrix)
        # add the minimum value of the fitnessvalues to a list
        Min_fitness_values.append(np.amin(fitnessValues))
        # add the mean value of the fitnessvalues to a list
        Mean_fitness_values.append(np.mean(fitnessValues))
        # take the best sequence from the population and add it to the fitest population
        fit_pop[trial] = population[list(fitnessValues).index(np.min(fitnessValues))]
            
    # Graph to plot the mean and minimum fitness values or the minimumm and mean value for each generation for each trial of the GA
    title = "Optimization of TSP \n population: {}  Probability of mutation: {} \n Number of Generations: {}  Crossover: {} \n Minimum Distance: {}Km Mean Distance: {:0.2f}Km".format(populationSize, probmutation, numberOfGenerations, crossovertype, np.amin(Min_fitness_values),np.mean(Mean_fitness_values))
    plt.title(title)
    ylim = (0, max(Mean_fitness_values) / 100 * 120)
    plt.ylim(ylim)  
    #plt.xlabel("Number of generations") 
    plt.xlabel("GA Trial Number")
    plt.ylabel("Distance (Km)")
    xaxis = np.arange(0,len(Min_fitness_values), 1)
    plt.grid()
    plt.plot(xaxis, Min_fitness_values,  color="r", label = "Minimum  Distance")
    plt.plot(xaxis, Mean_fitness_values,  color="g", label = "Mean Distance")
    plt.legend(loc="best")
    plt.show()

    # printing the best sequence or sequences of the trial
    Min_fitness_values = np.asarray(Min_fitness_values)  
    print("Minimum sequence values:")
    for i in range(0, len(Min_fitness_values)):
        if Min_fitness_values[i] == np.min(Min_fitness_values):
                print(fit_pop[i])
                
    # histogram of the distribution of minimum values         
    title = "Distribution of Optimal Sequences \n population: {}  Probability of mutation: {} \n Number of Generations: {}  Crossover: {} \n Minimum Distance: {}Km Mean Minimum Distance: {:0.2f}Km".format(populationSize, probmutation, numberOfGenerations, crossovertype, np.amin(Min_fitness_values),np.mean(Min_fitness_values))           
    plt.hist(Min_fitness_values)
    plt.title(title)
    plt.xlabel("Distance (Km)")
    plt.ylabel("Frequency")    
main()
    
    
    