import numpy as np


def cot(tach, Fs_tach, ppr, trigger, vibration, Fs_vibration, orders):
    # tach              = Tachometer signal (1D)
    # Fs_tach           = Tachometer sampling frequency
    # ppr               = Pulses per revolution (Tachometer)
    # trigger           = Tachometer signal voltage at pulse
    # vibration         = Vibration signal (1D)
    # Fs_vibration      = Vibration signal sampling frequency
    # Orders            = No. of Orders (int)

    # Define tacho time array
    dt_tach = 1/Fs_tach # Tacho signal time step
    N_tach = np.size(tach) # No. of Samples for tacho
    tf_tach = dt_tach * N_tach # End time of tacho signal
    t_tach = np.linspace(0, tf_tach, N_tach) # Tacho signal time array

    # Find time points/array indices of tacho pulses
    pulse_ind = np.where(tach == trigger)
    pulse_ind = pulse_ind[0] # Reshape

    # Get time points at each PPRth Pulse
    tf_window = t_tach[pulse_ind[ppr-1]]    # Time at first PPRth pulse
    N_windows = int(len(pulse_ind)/ppr)     # No. of windows
    ppr_ind = np.zeros(N_windows)           # PPRth index storage array
    Nr = N_windows                          # No. of revolutions

    # Loop through revolutions to get PPRth indices
    for i in range(1, N_windows+1):
      ppr_ind[i-1] = pulse_ind[ppr*i - 1]

    ppr_ind = ppr_ind[ppr_ind != 0] # Remove zero indices
    ppr_ind = ppr_ind.astype(int) # Assign to integers
    t_tach_ppr_ind = t_tach[ppr_ind] # Get tacho time points at PPRth indices

    # Correlate PPRth time points from tacho signal with vibration signal
    dt_sig = 1/Fs_vibration # Vibration signal time step
    N_sig = np.size(vibration) # No. of samples for vib.
    tf_sig = dt_sig * N_sig # End time of vib. signal
    t_sig = np.linspace(0, tf_sig, N_sig) # Vib. signal time array
    t_sig_ppr_ind = np.zeros(len(ppr_ind)) # PPRth time point storage array

    # Define seeking function
    def seek(x, array):
      value = array[0]

      for i in range(len(array)):
        if abs(x-array[i]) < abs(x-value):
          value = array[i]
      return value

    # Determine closest time vib. signal to tacho signal
    for i in range(0, len(ppr_ind)):
      t_sig_ppr_ind[i] = seek(t_tach_ppr_ind[i], t_sig)

    t_sig_ind = np.zeros(len(t_sig_ppr_ind)) # Time points storage array

    # Determine indices of PPRth times of vib. signal
    for i in range(0, len(t_sig_ppr_ind)):
      ind_check = np.where(t_sig == t_sig_ppr_ind[i])
      t_sig_ind[i] = ind_check[0]

    t_sig_ind = t_sig_ind.astype(int) # Assign to integers

    # Interpolate order=N points between rotations
    N = orders
    t_cot = []
    sig_cot = []

    # Perform COT for full rotations (Nr-1)
    for i in range(0, len(t_sig_ppr_ind)-1):

      t_start = t_sig[t_sig_ind[i]]
      t_end = t_sig[t_sig_ind[i+1]]

      t_rsmp = np.linspace(t_start, t_end, N)
      sig_int = np.interp(t_rsmp, t_sig, vibration)

      t_cot.append(t_rsmp)
      sig_cot.append(sig_int)

    t_cot = np.array(t_cot).reshape(-1)
    sig_cot = np.array(sig_cot).reshape(-1) # Change this function to average out the values to make one function for tsa & cot

    return t_cot, sig_cot
