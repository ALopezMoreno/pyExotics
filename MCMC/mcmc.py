# Class for doing MCMC inference on oscillation parameters using
# usual pdg priors
import numpy as np
from vanilla import oscillatorBase
from nonUnitary import sterileOsc
import time
from tqdm import tqdm

#  --- CLASSES FOR DOING MCMC ON PROBABILITIES ---  #
class MCMC_toy_experiment:
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
        self.therewasupdate=False
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
        if not (0 < self.sin2th12_prop < 1):
            return False
        if not (0 < self.sin2th23_prop < 1):
            return False
        if not (0 < self.sin2th13_prop < 1):
            return False
        if self.dcp_prop > np.pi:
            self.dcp_prop += -2*np.pi
        elif self.dcp_prop < -np.pi:
            self.dcp_prop += 2*np.pi
        return True

    def get_logLikelihood(self):
        #  Update parameter values in the oscillator class:
        if not self.checkBounds():
            self.logL = 999999999999999
            return 0

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

        return 1

    def propose_step(self): #FIX THIS LOOKING AT COVARIANCE!!!!!
        self.sin2th12_prop = np.random.normal(self.sin2th12, 5 * self.stepsize * np.sqrt(0.0001689))#4 * self.stepsize)
        self.sin2th23_prop = np.random.normal(self.sin2th23, 5 * self.stepsize * np.sqrt(0.000441))
        self.sin2th13_prop = np.random.normal(self.sin2th13, 30 * self.stepsize * np.sqrt(4.9*10**(-7)))#* 0.6 * self.stepsize)
        self.dcp_prop = np.random.normal(self.dcp, 0.05 * self.stepsize*np.sqrt(39.48))#35 * self.stepsize)

    def accept_step(self):
        self.sin2th12 = self.sin2th12_prop
        self.sin2th23 = self.sin2th23_prop
        self.sin2th13 = self.sin2th13_prop
        self.dcp = self.dcp_prop
        self.accepted += 1
        self.therewasupdate=True
    def update_step(self):
        start_time = time.time()
        self.therewasupdate=False
        old_logL = self.logL
        self.logL = 0
        self.propose_step()
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

# --- CLASS FOR CALCULATING THE STERILE REACTOR PRIOR --- #
class MCMC_toy_reactor_prior(MCMC_toy_experiment):
    def __init__(self, P_reactor, stepsize, sterile=False, startPoint=None):
        self.P = P_reactor #gaussian [mean, standard_deviation]
        self.stepsize = stepsize
        self.sterile = sterile
        self.E = 0.6
        self.L = 295
        self.time=0
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

        if self.sterile:
            self.osc = sterileOsc.oneSterile_oscillator(self.L, self.E)
            # we don't have initial guesses for the sterile parameters
            self.sin2th14 = np.random.uniform(0, 0.1)
            self.sin2th24 = np.random.uniform(0, 1)
            self.sin2th34 = np.random.uniform(0, 1)
            self.phi14 = np.random.uniform(-np.pi, np.pi)
            self.phi24 = np.random.uniform(-np.pi, np.pi)
            self.phi34 = np.random.uniform(-np.pi, np.pi)

            self.sin2th14_prop = self.sin2th14
            self.sin2th24_prop = self.sin2th24
            self.sin2th34_prop = self.sin2th34
            self.phi14_prop = self.phi14
            self.phi24_prop = self.phi24
            self.phi34_prop = self.phi34

        else:
            self.osc = oscillatorBase.Oscillator(self.L, self.E)

        #  remaining MCMC stuff
        self.therewasupdate=False
        self.stepN = 1
        self.accepted = 0
        self.acceptance_rate = 0
        #get LogL of initial step
        self.get_logLikelihood()

    def propagate(self):
        return self.osc.getOsc(0, 0, antineutrino=True)

    # We define a new logLikelihood which only takes P_ee into account
    def get_logLikelihood(self):
        if self.sterile:
            if not self.checkBounds_sterile():
                self.logL = 999999999999999
                return 0

        if not self.checkBounds():
            self.logL = 999999999999999
            return 0

        #  Update parameter values in the oscillator class:
        self.logL = 0
        self.osc.theta12 = np.arcsin(np.sqrt(self.sin2th12_prop))
        self.osc.theta23 = np.arcsin(np.sqrt(self.sin2th23_prop))
        self.osc.theta13 = np.arcsin(np.sqrt(self.sin2th13_prop))
        self.osc.dcp = self.dcp_prop
        self.osc.setPMNS()
        if self.sterile:
            self.osc.theta14 = np.arcsin(np.sqrt(self.sin2th14_prop))
            self.osc.theta24 = np.arcsin(np.sqrt(self.sin2th24_prop))
            self.osc.theta34 = np.arcsin(np.sqrt(self.sin2th34_prop))
            self.osc.phi14 = self.phi14_prop
            self.osc.phi24 = self.phi24_prop
            self.osc.phi34 = self.phi34_prop
            self.osc.set_nonUnitary()
            self.osc.build_mixing_matrix()

        #  Calculate probabilities for the proposed steps with smearing according to the appropriate energy profile
        self.P_ee_prop = self.propagate()

        #  Compare with the "data" probabilities and draw a likelihood from it
        self.logL = 1/2 * ( (self.P_ee_prop - self.P[0]) / self.P[1] )**2


    def propose_step_sterile(self): #FIX THIS LOOKING AT COVARIANCE!!!!!
        self.sin2th14_prop = np.random.normal(self.sin2th14, 10 * self.stepsize * np.sqrt(4.9*10**(-7)))
        self.sin2th24_prop = np.random.normal(self.sin2th24, 1 * self.stepsize * np.sqrt(0.000441))
        self.sin2th34_prop = np.random.normal(self.sin2th34, 1 * self.stepsize * np.sqrt(0.000441))
        self.phi14_prop = np.random.normal(self.phi14, 0.05 * self.stepsize*np.sqrt(39.48))
        self.phi24_prop = np.random.normal(self.phi24, 0.05 * self.stepsize*np.sqrt(39.48))
        self.phi34_prop = np.random.normal(self.phi34, 0.05 * self.stepsize*np.sqrt(39.48))

    def accept_step_sterile(self):
        self.sin2th14 = self.sin2th14_prop
        self.sin2th24 = self.sin2th24_prop
        self.sin2th34 = self.sin2th34_prop
        self.phi14 = self.phi14_prop
        self.phi24 = self.phi24_prop
        self.phi34 = self.phi34_prop

    def checkBounds_sterile(self):
        if not (0 < self.sin2th14_prop < 1):
            return False
        if not (0 < self.sin2th24_prop < 1):
            return False
        if not (0 < self.sin2th34_prop < 1):
            return False

        if self.phi14_prop > np.pi:
            self.phi14_prop += -2*np.pi
        elif self.phi14_prop < -np.pi:
            self.phi14_prop += 2*np.pi

        if self.phi24_prop > np.pi:
            self.phi24_prop += -2*np.pi
        elif self.phi24_prop < -np.pi:
            self.phi24_prop += 2*np.pi

        if self.phi34_prop > np.pi:
            self.phi34_prop += -2*np.pi
        elif self.phi34_prop < -np.pi:
            self.phi34_prop += 2*np.pi
        return True
    def update_step(self):
        start_time = time.time()
        self.therewasupdate=False
        old_logL = self.logL
        self.logL = 999
        self.propose_step()
        if self.sterile:
            self.propose_step_sterile()
        self.get_logLikelihood()
        ratio = np.exp(old_logL - self.logL)

        if ratio >= 1:
            self.accept_step()
            if self.sterile:
                self.accept_step_sterile()
        else:
            throw = np.random.uniform(0, 1)
            if throw <= np.abs(ratio):
                self.accept_step()
                if self.sterile:
                    self.accept_step_sterile()
            else:
                self.logL = old_logL

        self.stepN += 1

        self.acceptance_rate = self.accepted / self.stepN
        self.time = time.time() - start_time

    def runChain(self, nsteps, name, start=False):
        pbar = tqdm(total=nsteps)
        start_time = time.time()
        if self.sterile:
            step = ", ".join(map(str, [self.sin2th12, self.sin2th23, self.sin2th13, self.dcp,
                                       self.sin2th14, self.sin2th24, self.sin2th34, self.phi14,
                                       self.phi24, self.phi34, self.logL,
                                       self.acceptance_rate, self.time, self.stepN]))
        else:
            step =", ".join(map(str, [self.sin2th12, self.sin2th23, self.sin2th13, self.dcp, self.logL, self.acceptance_rate, time.time() - start_time, self.stepN]))
        if start == True:
            f = open(name, "a")
        else:
            if self.sterile:
                titles = ", ".join(map(str, ['sin2th12', 'sin2th23', 'sin2th13', 'dcp',
                                             'sin2th14', 'sin2th24', 'sin2th34', 'ph14',
                                             'phi24', 'phi34', 'logL', 'acceptance_rate', 'steptime', 'step']))
            else:
                titles = ", ".join(map(str, ['sin2th12', 'sin2th23', 'sin2th13', 'dcp', 'logL', 'acceptance_rate',
                                             'steptime', 'step']))

            f = open(name, "x")
            f.write(titles + '\n')
            f.write(step + '\n')
        while self.accepted <= nsteps:
            self.update_step()
            if self.sterile:
                step = ", ".join(map(str, [self.sin2th12, self.sin2th23, self.sin2th13, self.dcp,
                                           self.sin2th14, self.sin2th24, self.sin2th34, self.phi14,
                                           self.phi24, self.phi34, self.logL,
                                           self.acceptance_rate, self.time, self.stepN]))
            else:
                step = ", ".join(map(str,[self.sin2th12, self.sin2th23, self.sin2th13, self.dcp, self.logL, self.acceptance_rate, self.time, self.stepN]))
            if self.therewasupdate:
                f.write(step + '\n')
                #if self.stepN%1000 == 0:
                #    print("step %i of %i took %f" % (self.stepN, nsteps, self.time))
                #    #print(step)
                pbar.update(1)
        pbar.close()
        f.close()