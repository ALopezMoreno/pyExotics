# Experimental setups for running these toy models
import numpy as np
import sys
sys.path.append("./")
class experiment():
    #  Propagate probabilities for a given energy profile and experimental setup

    def __init__(self, eprofile, baseline, matter=False, smooth=False):
        self.energies = eprofile
        self.L = baseline
        self.matter_effects = matter
        self.smooth= smooth

    def propagate(self, oscillator, nu_alpha, nu_beta, antineutrino=False):
        # use an oscillator class to propagate the energy profile
        self.osc = oscillator
        self.osc.block = True
        # we don't smear, we calculate bin-centre energy and propagate bin values
        self.osc.nsmear = 0
        self.osc.Esmear = 0

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

            P = np.zeros(len(Er))

            if self.smooth != False:

                for i in range(len(E)):
                    P[i] = 0
                    tempE = np.linspace(Eb[i], Eb[i+1], self.smooth+1)[1:]
                    for j in tempE:
                        self.osc.update(self.L, j)
                        P[i] += self.osc.getOsc(nu_alpha, nu_beta, antineutrino=antineutrino) / self.smooth
            else:
                for i in range(len(E)):
                    self.osc.update(self.L, E[i])
                    P[i] = self.osc.getOsc(nu_alpha, nu_beta, antineutrino=antineutrino)

            oscillated_eprofile = np.multiply(P, Eh)
            return oscillated_eprofile



class eProfile():
    # NoVA and T2K-like profiles refer to spectra which correspond to
    # shifted and or normalised versions of the T2K and NOvA reconstructed energy distributions

    def __init__(self, etype):
        self.type = etype
        nova = "NOvA"
        t2k = "T2K"
        if self.type.casefold() == t2k.casefold():
            self.T2Klike()
        elif self.type.casefold() == nova.casefold():
            self.NOvAlike()
        else:
            print("ERROR: DID NOT RECOGNISE ENERGY TYPE. DO YOU MEAN T2K OR NOvA?")
            exit()

    def T2Klike(self):
        # Load T2K near detector flux predictions (-ve horn)
        fluxData = np.loadtxt(
            "../fluxes/T2Kflux2020/t2kflux_2020_public_release/t2kflux_2020_plus250kA_nominal_nd280.txt", skiprows=3, dtype='str')
        binEdges = np.append(fluxData[:, 1].astype(float), 30.00)
        self.numuBE = binEdges
        self.numu_barBE = binEdges
        self.nueBE = binEdges
        self.nue_barBE = binEdges

        self.numu = fluxData[:, 4].astype(float)
        self.numu_bar = fluxData[:, 5].astype(float)
        self.nue = fluxData[:, 6].astype(float)
        self.nue_bar = fluxData[:, 7].astype(float)

    def NOvAlike(self):
        # Load NOvA near detector flux predictions (FHC)
        numu = np.loadtxt("fluxes/NOvAflux/numu.txt", skiprows=1, dtype='float')
        numu_bar = np.loadtxt("fluxes/NOvAflux/numubar.txt", skiprows=1, dtype='float')
        nue = np.loadtxt("fluxes/NOvAflux/nue.txt", skiprows=1, dtype='float')
        nue_bar = np.loadtxt("fluxes/NOvAflux/nuebar.txt", skiprows=1, dtype='float')

        self.numuBE = np.append(numu[:, 0], numu[-1, 1])
        self.numu_barBE = np.append(numu_bar[:, 0], numu_bar[-1, 1])
        self.nueBE = np.append(nue[:, 0], nue[-1, 1])
        self.nue_barBE = np.append(nue_bar[:, 0], nue_bar[-1, 1])

        self.numu = numu[:, 2]
        self.numu_bar = numu_bar[:, 2]
        self.nue = nue[:, 2]
        self.nue_bar = nue_bar[:, 2]
