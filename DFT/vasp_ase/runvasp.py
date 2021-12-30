#!/usr/bin/env python

from ase.io.vasp import read_vasp
from pyDFTutils.perovskite.cubic_perovskite import gen_primitive
import os
from spglib import spglib
import numpy as np
from math import pi
from pyDFTutils.ase_utils import *
from pyDFTutils.vasp.myvasp import myvasp, default_pps
from pyDFTutils.vasp.vasp_utils import read_efermi
from multiprocessing import Process, Pool
from pyDFTutils.wannier90.wannier import WannierInput, run_wannier
from pyDFTutils.perovskite.frozen_mode import gen_P21c_perovskite


def gen_atoms(amp_slater=0.0, amp_rot=0.0):
    atoms = gen_P21c_perovskite(name='SrMnO3', cell=[3.81, 3.81, 3.81],
                                supercell_matrix=[
        [0, 1, 1], [1, 0, 1], [1, 1, 0]],
        modes=dict(
        # R2_m_O1=0.1, #breathing
        # R3_m_O1=1.0,
        # R3_m_O2=1.0,  # R3-[O1:c:dsp]A2u(b), O, out-of-plane-stagger, inplane antiphase
        # R5_m_O1=2.0,  # R5-[O1:c:dsp]Eu(a), O a-
        # R5_m_O2=2.0,  # R5-[O1:c:dsp]Eu(a), O b-
        R5_m_O3=amp_rot,  # R5-[O1:c:dsp]Eu(c), O  c-
        # X5_m_A1=1.0,  # [Nd1:a:dsp]T1u(a), A , Antiferro mode

        # M2_p_O1=2.0,  # M2+[O1:c:dsp]Eu(a), O, In phase rotation c+

        # M3_p_O1=0.1,  # M3+[O1:c:dsp]A2u(a), O, D-type JT inplane stagger

        # M5_p_O1=1.0,  # M5+[O1:c:dsp]Eu(a), O, Out of phase tilting

        # M4_p_O1=1.0 , # M4+[O1:c:dsp]A2u(a), O, in-plane-breathing (not in P21/c)
        G_Ax=0.0,
        G_Ay=0.0,
        G_Az=0.0,
        G_Sx=0.0,
        G_Sy=0.0,
        G_Sz=amp_slater,
        G_Axex=0.0,
        G_Axey=0.0,
        G_Axez=0.0,
        G_Lx=0.0,
        G_Ly=0.0,
        G_Lz=0.0,
        G_G4x=0.0,
        G_G4y=0.0,
        G_G4z=0.0,
    )
    )
    #write('P4mm.vasp', atoms, vasp5=True)
    # vesta_view(atoms)
    return atoms


def test_var(amp_slater=0.0, ecut=500, nk=5, U=2):
    atoms = gen_atoms(amp_slater=amp_slater)
    atoms.set_pbc(True)

    # adjust atoms structure
    # atoms=set_substrate(atoms,a,a,m=np.sqrt(2))
    # atoms.rattle()
    # atoms.rotate('z',pi/4,rotate_cell=True)

    # Set magnetic
    #mag = np.array([0, 0, 2, 2])
    atoms = set_element_mag(atoms, 'Mn', [3, -3])
    print(atoms.get_initial_magnetic_moments())

    # basic settings: XC, setups, kpts
    mycalc = myvasp(
        xc='PBE',
        gga='PS',
        setups=default_pps,
        ispin=2,
        icharg=2,
        kpts=[6, 6, 6],
        gamma=True,
        prec='normal',
        istart=1,
        lmaxmix=4,
        encut=520)
    mycalc.set(lreal=False, kpar=1, ncore=1, algo='normal')

    # electronic
    mycalc.set(ismear=-5, sigma=0.1, nelm=100, nelmdl=-6, ediff=1e-7)

    # LDA+U

    # structure relaxation
    mycalc.set(
        nsw=100,
        ibrion=1,
        isif=3,
        addgrid=True,
        ediffg=-1e-3,
        potim=0.28,
        smass=1.2,
    )

    # SET
    atoms.set_calculator(mycalc)

    # Relax
    mycalc.set(
        ldau=True,
        ldautype=1,
        ldau_luj={
            'Mn': {
                'L': 2,
                'U': 3,
                'J': 0
            }, }
    )

    mycalc.set(icharg=0, nsw=100, ismear=-5, nelmdl=5)
    # mycalc.clean()
    # atoms=mycalc.myrelax_calculation(pre_relax=False)

    nbands = 72
    mycalc.set(icharg=0, nsw=0, ismear=-5, nelmdl=-3, nbands=nbands)
    #mycalc.scf_calculation(atoms, ismear=-5)
    # mycalc.ldos_calculation(atoms)
    # return
    mycalc.set(lwannier90=True, lwrite_unk=False, lwrite_mmn_amn=True, npar=8)
    wa = WannierInput(atoms=atoms)
    # efermi = read_efermi()  # 5.05
    efermi = 4.0
    wa.set(mp_grid=[6, 6, 6], num_bands=nbands, guiding_centres=False,
           num_iter=300, kmesh_tol=1e-6, search_shells=24,)
    wa.add_basis('O', orb='p')
    wa.add_basis('Mn', orb='d')
    wa.set_energy_window([-12, 7], [-8, 4], shift_efermi=efermi)
    wa.set(write_xyz=True,  write_hr=True)
    wa.set_kpath(np.array([[0.0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [
                 0, 0, 1]])/2, ['GM', 'Z', 'A', 'R', 'Z'], npoints=410)
    #wa.set(wannier_plot_format='xcrysden', wannier_plot_mode='crystal', wannier_plot=True,wannier_plot_supercell=2)
    # wa.set_kpath(([0.0,0,1],[0,0,0],[0.5,0,0]),['Z','GM','M'],npoints=410)
    wa.set(bands_plot_format='gnuplot', bands_plot=True)
    wa.write_input()
    mycalc.scf_calculation(atoms)
    try:
        print("Runing wannier")
        run_wannier(spin='up')
        run_wannier(spin='dn')
    except Exception:
        pass


def func(amp_slater):
    dir_name = f'U3_SrMnO3_111_slater{amp_slater:.2f}'
    cur_dir = os.getcwd()
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.chdir(dir_name)
    test_var(amp_slater=amp_slater)
    os.chdir(cur_dir)


def test_par():
    amp_list = [-0.1, 0.0, 0.1]
    for a in amp_list:
        func(a)


if __name__ == '__main__':
    test_par()
