from imutils.video import FPS
from scipy.optimize import basinhopping
from numpy.random import rand
import os
import utils

class TubeShifting():
    """
    Handler class for search optimization using basin hopping and "L-BFGS-B"
    for tube shifting and rearrangement
    """    
    def __init__(self, seq_to_optimize, iterations, stepsize):
        """Initialize class parameters

        Args:
            energy_function (method): energy function initialization method
            iterations (int): number of optimization iterations
            stepsize (float): maximum stepsize
        """        
        self.f = self.energy_function
        self.__set_properties(iterations, stepsize)
        self._tubes = seq_to_optimize
    
    def __set_properties(self, iterations, stepsize):
        """Sets the optimization configurations

        Args:
            iterations (int): number of optimization iterations
        """        
        self._method = "L-BFGS-B"
        self._niter  = iterations
        self._disp   = True
        self._stepsize = stepsize
 
    def start(self):
        """Performs the global optimization

        Args:
            seq_to_optimize (string): path to tubes

        Returns:
            list: optimization solution
        """
        nbrOfTubes = len(os.listdir(self._tubes))
        print(f"[INFO] Staring Basin Hopping optimization of {nbrOfTubes} tubes")        
        # Initial parameters
        pt = [rand(nbrOfTubes) * 0]
        # Set constraints on the optimization 
        constraints={"fun": self.constraint_positive, "type": "ineq"}
        bounds = tuple(zip((0.0,)*nbrOfTubes, (None,)*nbrOfTubes))
        minimizer_kwargs = {"method": self._method, "constraints": constraints, "bounds": bounds}
        # Perform basin hopping global optimization
        result = basinhopping(self.f, pt, stepsize=self._stepsize, niter=self._niter, disp=self._disp, minimizer_kwargs=minimizer_kwargs)
        # Summarize the result
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        # Evaluate solution
        solution = result['x']
        evaluation = self.f(solution)
        print('Solution: f(%s) = %.5f' % (solution, evaluation))
        return [int(round(point, 0)) for point in solution]
    
    def constraint_positive(x):
        """Checks if the point fullfils set constraints

        Args:
            x (list): list of points

        Returns:
            bool: whether the point fullfils the constraints or not
        """        
        if (x < 0).any():
            return -1
        else:        
            return 1 

    def energy_function(self, pt):
        """Calculates the energy for a given set of start times

        Args:
            pt (list): start times of the tubes

        Returns:
            float: solution given the input start times
        """    
        pt = [int(round(point, 0)) for point in pt]
        overall_energy = 0
        tubes = [int(x) for x in os.listdir(self._tubes)]

        # Getting m(s, e), the set of start and end times for each tube
        m = []
        for index, tube in enumerate(tubes):
            S = pt[index]
            E = S + len(os.listdir(os.path.join(self._tubes, str(tube)))) -1
            m.append([S, E])
        # Calculating the energy per tube
        for index, tube1 in enumerate(tubes):
            for index1, tube2 in enumerate(tubes[index+1:]):
                print(f"[INFO] Calculating Ec(m{tube1}, m{tube2})")
                #Get time overlap between both tubes
                Si, Sj, Ei, Ej = m[index][0], m[index+index1+1][0], m[index][1], m[index+index1+1][1]
                Mi = range(Si, Ei)
                Mj = range(Sj, Ej)
                tau = set(Mi).intersection(Mj)
                # Calculate overlap between the two tubes
                # for each frame they overlap in
                for frame in tau:
                    overall_energy += utils.overlap_masks(str(tube1), str(tube2), Mi.index(frame), Mj.index(frame))
        return overall_energy
    
if __name__ == "__main__":
    optimize = TubeShifting("tubes/", 20, 3.0)
    solution = optimize.start()
    print(solution)