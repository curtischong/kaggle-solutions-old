- a type of feature used in speech and audio signal processing
	- speech recognition, speaker identification, and music genre classification.
- GPT:
	- The process of calculating MFCCs involves a few important stages:
	1. **Frame the signal**: The continuous audio signal is divided into short frames (typically 20-40 milliseconds).
	2. **Apply a window function to each frame**: This is typically a Hamming window, which is used to minimize the discontinuities at the beginning and end of each frame.
	3. **Perform Fourier transform**: This converts the signal from the time domain to the frequency domain, producing the spectrum of the signal.
	4. **Apply the Mel filterbank to the power spectra, and sum the energy in each filter**: This stage emphasizes frequencies that are important for human hearing. The Mel scale simulates the nonlinear human ear perception of sound, as humans are better at detecting differences in lower frequencies than high.
	5. **Take the logarithm of all filterbank energies**: As human perception of sound intensity also follows a logarithmic scale, this step more closely approximates human auditory perception.
	6. **Take the Discrete Cosine Transform (DCT) of the log filterbank energies**: This smooths the frequency spectrum and results in a set of Mel-Frequency Cepstral Coefficients. The DCT also helps to de-correlate the energies in the different filters.