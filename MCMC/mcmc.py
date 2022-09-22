# Class for doing MCMC inference on oscillation parameters using
# usual pdg priors
import numpy as np
from vanilla import oscillatorBase
import time

class MCMC:
    def __init__(self, Pee, Pemu, Pmumu, PemuBar, eProfile, baseline, stepsize, startPoint=None):
        # Four channels are 1) e  survival
        #                   2) mu survival
        #                   3) e->mu neutrino
        #                   4) e->mu antineutrino
        self.P1 = Pee
        self.P2 = Pmumu
        self.P3 = Pemu
        self.P4 = PemuBar

        self.energies = eProfile
        self.L = baseline
        #  use an oscillator class to propagate and get likelihood  - dummy inputs for now
        self.osc = oscillatorBase.Oscillator(self.L, 1, smearing=[0, 0])
        self.osc.block = True

        self.stepsize = stepsize #  size of steps for new draws

        #  ---- init oscillation parametres ----  #
        if startPoint:
            self.sin2th12 = startPoint[0]
            self.sin2th23 = startPoint[1]
            self.sin2th13 = startPoint[2]
            self.dcp = startPoint[3]
        else:
            self.sin2th12 = np.random.uniform(0, 1)
            self.sin2th23 = np.random.uniform(0, 1)
            self.sin2th13 = np.random.uniform(0, 1)
            self.dcp = np.random.uniform(-np.pi, np.pi)

        self.sin2th12_prop = self.sin2th12
        self.sin2th23_prop = self.sin2th23
        self.sin2th13_prop = self.sin2th13
        self.dcp_prop = self.dcp

        #  remaining MCMC stuff
        self.stepN = 1
        self.accepted = 0
        self.acceptance_rate = 0
        #get LogL of initial step
        self.get_logLikelihood()

    #  Function for running an MCMC chain
    def runChain(self, nsteps, name, start=False):
        start_time = time.time()
        step =", ".join(map(str, [self.sin2th12, self.sin2th23, self.sin2th13, self.dcp, self.logL, self.acceptance_rate, time.time() - start_time, self.stepN]))
        if start == True:
            f = open(name, "a")
        else:
            titles = ", ".join(map(str,['sin2th12', 'sin2th23', 'sin2th13', 'dcp', 'logL', 'acceptance_rate', 'steptime', 'step']))
            f = open(name, "x")
            f.write(titles + '\n')
            f.write(step + '\n')
        while self.stepN <= nsteps:
            self.update_step()
            step = ", ".join(map(str,[self.sin2th12, self.sin2th23, self.sin2th13, self.dcp, self.logL, self.acceptance_rate, self.time, self.stepN]))
            f.write(step + '\n')
            if self.stepN%1000 == 0:
                print("step %i of %i took %f" % (self.stepN, nsteps, self.time))
                #print(step)
        f.close()


    #  --- FUNCTIONS FOR CALCULATING THE OSCILLATION PROBABILITY GIVEN THE PARAMS ---  #
    #  (stolen from the experiments class)

    def propagate(self, nu_alpha, nu_beta, antineutrino=False):
        # use an oscillator class to propagate

        if nu_alpha > 1:
            print("we have no support for tau channels in experimental setups")
            exit()
        else:
            # get bin counts and edges, then calculate bin centres
            if nu_alpha == 0:
                if antineutrino == 0:
                    Eh = self.energies.nue
                    Eb = self.energies.nueBE
                else:
                    Eh = self.energies.nue_bar
                    Eb = self.energies.nue_barBE
            else:
                if antineutrino == 0:
                    Eh = self.energies.numu
                    Eb = self.energies.numuBE
                else:
                    Eh = self.energies.numu_bar
                    Eb = self.energies.numu_barBE
            Er = Eb[1:] - Eb[:-1]
            E = Eb[:-1] + Er / 2
            Ptemp = 0

            #Sum over all energies
            for i in range(len(E)):
                self.osc.Esmear = 0
                self.osc.nsmear = 0
                self.osc.update(self.L, E[i])
                Ptemp += self.osc.getOsc(nu_alpha, nu_beta) * Eh[i] * Er[i]
            P = Ptemp / np.sum(np.multiply(Eh, Er))
            return P


    #  --- FUNCTIONS REGARDING THE METROPOLIS HASTINGS MCMC ---  #
    def checkBounds(self):
        while not (0 < self.sin2th12_prop < 1):
            self.sin2th12_prop = np.random.normal(self.sin2th12, self.stepsize)
        while not (0 < self.sin2th23_prop < 1):
            self.sin2th23_prop = np.random.normal(self.sin2th23, self.stepsize)
        while not (0 < self.sin2th13_prop < 1):
            self.sin2th13_prop = np.random.normal(self.sin2th13, self.stepsize)
        if self.dcp_prop > np.pi:
            self.dcp_prop += -2*np.pi
        elif self.dcp_prop < -np.pi:
            self.dcp_prop += 2*np.pi

    def get_logLikelihood(self):
        #  Update parameter values in the oscillator class:
        self.logL = 0
        self.osc.theta12 = np.arcsin(np.sqrt(self.sin2th12_prop))
        self.osc.theta23 = np.arcsin(np.sqrt(self.sin2th23_prop))
        self.osc.theta13 = np.arcsin(np.sqrt(self.sin2th13_prop))
        self.osc.dcp = self.dcp_prop
        self.osc.setPMNS()

        #  Calculate probabilities for the proposed steps with smearing according to the appropriate energy profile
        self.P1_prop = self.propagate(0, 0)
        self.P2_prop = self.propagate(1, 1)
        self.P3_prop = self.propagate(1, 0)
        self.P4_prop = self.propagate(1, 0, antineutrino=True)

        #  Compare with the "data" probabilities and draw a likelihood from it
        self.logL += np.sqrt( (self.P1_prop - self.P1)**2 * 1 +
                       (self.P2_prop - self.P2)**2 * 1 +
                       (self.P3_prop - self.P3)**2 * 1 +
                       (self.P4_prop - self.P4)**2 * 1)

        # Apply solar constraint:
        self.logL += 0.5 * ((self.sin2th12_prop - 0.307) / np.sqrt(0.001689)) ** 2
        # Apply reactor constraints
        self.logL += 0.5 * ((self.sin2th13_prop - 0.0218) / np.sqrt(4.9*10**(-7))) ** 2

    def propose_step(self):
        self.sin2th12_prop = np.random.normal(self.sin2th12, 6 *0.25 * np.sqrt(0.001689))#4 * self.stepsize)
        self.sin2th23_prop = np.random.normal(self.sin2th23, 2.38 *0.25 * self.stepsize)
        self.sin2th13_prop = np.random.normal(self.sin2th13, 6 *0.25 * np.sqrt(4.9*10**(-7)))#* 0.6 * self.stepsize)
        self.dcp_prop = np.random.normal(self.dcp, 2.38 *0.25 * 3)#35 * self.stepsize)

    def accept_step(self):
        self.sin2th12 = self.sin2th12_prop
        self.sin2th23 = self.sin2th23_prop
        self.sin2th13 = self.sin2th13_prop
        self.dcp = self.dcp_prop
        self.accepted += 1

    def update_step(self):
        start_time = time.time()
        old_logL = self.logL
        self.logL = 0
        self.propose_step()
        self.checkBounds()
        self.get_logLikelihood()
        ratio = old_logL / self.logL

        if np.abs(ratio) >= 1:
            self.accept_step()

        else:
            throw = np.random.uniform(0, 1)
            if throw <= np.abs(ratio):
                self.accept_step()
            else:
                self.logL = old_logL

        self.stepN += 1


        self.acceptance_rate = self.accepted / self.stepN
        self.time = time.time() - start_time

