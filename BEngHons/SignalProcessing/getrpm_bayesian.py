import numpy as np
import scipy.signal as sig
import numpy.matlib as npml
from scipy.sparse.linalg import spsolve


def maketime(X, Fs):
    """
    X = Signal
    Fs = sampling frequency

    Returns:
        time
    """
    t0 = 0
    t1 = len(X) / Fs
    t = np.arange(t0, t1 + 1 / Fs, 1 / Fs)
    return t


def PerformBayesianGeometryCompensation(t, N, M, e=[], beta=10.0e10, sigma=10.0):
        '''
        PerformBayesianGeometryCompensation(t,N,M,e=[],beta=10.0e10,sigma=10.0)

        Perform geometry compensation on an incremental shaft encoder with N sections measured over M revolutions.

        Parameters
        ----------
        t :     1D numpy array of zeros crossing times.  The first zero crossing time indicates
                the start of the first section.  This array should therefore have exactly M*N + 1 elements.
        N:      The number of sections in the shaft encoder.
        M:      The number of complete revolutions over which the compensation must be performed.
        e:      An initial estimate for the encoder geometry.  If left an empty array, all sections are assumed equal.
        beta:   Precision of the likelihood function.
        sigma:  Standard deviation of the prior probability.

        Returns
        -------
        epost : An array containing the circumferential distances of all N sections.
        '''
        if len(t) != M * N + 1:
                print('Input Error: The vector containing the zero-crossing times should contain exactly N*M + 1 values')
                raise SystemExit
        if len(e) != 0 and len(e) != N:
                print('Input Error The encoder input should either be an empty list or a list with N elements')
                raise SystemExit
        # Initialize matrices
        A = np.zeros((2 * M * N - 1, N + 2 * M * N))
        B = np.zeros((2 * M * N - 1, 1))
        # Calculate zero-crossing periods
        T = np.ediff1d(t)

        # Insert Equation (11)
        A[0, :N] = np.ones(N)
        B[0, 0] = 2 * np.pi
        # Insert Equation (9) into A
        deduct = 0
        for m in range(M):
                if m == M - 1:
                    deduct = 1
                for n in range(N - deduct):
                    nm = m * N + n
                    A[1 + nm, n] = 3.0
                    A[1 + nm, N + nm * 2] = -1.0 / 2 * T[nm] ** 2
                    A[1 + nm, N + nm * 2 + 1] = -2 * T[nm]
                    A[1 + nm, N + (nm + 1) * 2 + 1] = -1 * T[nm]
        # Insert Equation (10) into A
        deduct = 0
        for m in range(M):
                if m == M - 1:
                    deduct = 1
                for n in range(N - deduct):
                    nm = m * N + n
                    A[M * N + nm, n] = 6.0
                    A[M * N + nm, N + nm * 2] = -2 * T[nm] ** 2
                    A[M * N + nm, N + (nm + 1) * 2] = -1 * T[nm] ** 2
                    A[M * N + nm, N + nm * 2 + 1] = -6 * T[nm]
        # Initialize prior vector
        m0 = np.zeros((N + 2 * M * N, 1))
        # Initialize and populate covariance matrix of prior
        Sigma0 = np.identity(N + 2 * M * N) * sigma ** 2
        # Populate prior vector
        if len(e) == 0:
                eprior = np.ones(N) * 2 * np.pi / N
        else:
                eprior = np.array(e) * 1.0
        m0[:N, 0] = eprior * 1.0
        for m in range(M):
                for n in range(N):
                    nm = m * N + n
                    m0[N + nm * 2 + 1, 0] = m0[n, 0] / T[nm]
        # Solve for mN (or x)

        l1 = len(A)
        l2 = len(A.T)
        l3 = len(B)

        SigmaN = Sigma0 + beta * A.T.dot(A)
        BBayes = Sigma0.dot(m0) + beta * A.T.dot(B)

        mN = np.array([spsolve(SigmaN, BBayes)]).T
        # Normalize encoder increments to add up to 2 pi
        epost = mN[:N, 0] * 2 * np.pi / (np.sum(mN[:N, 0]))
        # Return encoder geometry
        return epost


def getrpm(tacho, Fs, trig_level, slope, pprm, new_sample_freq):
    """
    1. tacho = Tachometer Signal
    2. Fs = Sampling Frequency in
    3. trig_level =  trigger level defined by author for a pulse
    4. slope = Positive or negative value for positive or negative pulses
    5. pprm = Tachometer pulses per revolution
    6. new_sample_freq = Reinterpolation sampling frequency

    NOTE! The trig function is very simple and basic which requires a
    clean tacho signal. In some cases, a filtered tacho may work better
    than the original one.

    See also SMOOTHRPM

    A simple smoothing is performed on the rpm signal. A harder smoothing
    may in some circumstances be required.

    Returns:
        TimeRPM, RPM

    Copyright (c) 2003-2006, Axiom EduTech AB, Sweden. All rights reserved.
    URL: http://www.vibratools.com Email: support@vibratools.com
    Revision: 1.1  Date: 2003-08-06
    Revision history
    2006-05-03      Added extrapolation in the interp1 call was added to
                    avoid NaN's.
    """

    if type(tacho) == list:
        tacho = np.array(tacho)

    y = np.sign(tacho - trig_level)
    dy = np.diff(y)

    tt = maketime(dy, Fs)

    if slope > 0:
        pos = np.nonzero(dy > 0.8)

    if slope < 0:
        pos = np.nonzero(dy < -0.8)

    yt = tt[pos]

    dt = np.diff(yt)
    dt = np.hstack([dt, np.array([dt[-1]])])

    t0 = 0
    t1 = len(dy) / Fs
    t_eval = np.arange(t0, t1 + 1 / Fs, 1 / Fs)
    cross_ind = np.where(tacho == trig_level)  # Tacho crosses
    cross_ind = cross_ind[0]
    t_cross = t_eval[cross_ind]
    n_revs = int(len(cross_ind) / pprm)
    eval_ind = (n_revs * pprm) + 1
    t_sections = t_cross[:eval_ind]

    spacing_rev = PerformBayesianGeometryCompensation(t_sections, pprm, n_revs, e=[], beta=10.0e10, sigma=10.0)

    spacing_store = np.zeros(len(dt))
    s_cnt = 0

    for i in range(0, len(dt) - 1):

        spacing_store[i] = spacing_rev[s_cnt]

        if s_cnt == 59:
            s_cnt = 0
        else:
            s_cnt += 1

    spacing = spacing_store

    rpm = (60 / (2 * np.pi)) * (spacing / dt)
    b = [0.25, 0.5, 0.25]
    a = 1
    rpm = sig.filtfilt(b, a, rpm)

    N = np.max(tt) * (new_sample_freq) + 1

    trpm = np.linspace(0, np.max(tt), int(N))

    rpm = np.interp(trpm, yt, rpm)
    pos = []
    cnt = 0
    for i in np.isnan(rpm):
        if i == False:
            pos.append(cnt)
        cnt += 1

    return trpm[pos], rpm[pos], spacing_store
